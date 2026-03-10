use std::sync::Arc;

use axum::{
    Form, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use clap::Parser;
use moka::future::Cache;
use ort::{
    execution_providers::{CUDAExecutionProvider, ROCmExecutionProvider, CPUExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},

    value::{Tensor, ValueType},
    tensor::TensorElementType,

};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::trace::{self, TraceLayer};
use tracing::Level;



type Model = Session;

#[derive(Parser)]
#[command(name = "ONNX-Service")]
#[command(version, about)]
struct Args {
    #[arg(long)]
    enable_logging: bool,
    #[arg(short, long, default_value_t = 3000)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    // As we can only run a single inference through each model, we need to mutex it
    model_cache: Cache<String, Arc<Mutex<Session>>>,
    // Cache the mutable, rolling input per model_url (overwritten on init=true)
    input_cache: Cache<String, Arc<Mutex<CachedInput>>>,
}

#[derive(Debug, Clone)]
struct CachedInput {
    data: Vec<f32>,
    sensors: usize,
    covariates: usize,
    input_size: usize,
}

// MAX_CACHED_MODELS needs to be > 0
static MAX_CACHED_MODELS: u64 = 8;

#[tokio::main]
async fn main() {
    // Consider configuration file if possible
    // In any case: Command line settings overwrite config file settings
    log_available_providers();
    let config = Args::parse();
    let service_builder = ServiceBuilder::new();
    let trace_layer = if config.enable_logging {
        let filter = tracing_subscriber::EnvFilter::new("INFO")
            // For ort crate only log errors
            .add_directive("ort=error".parse().unwrap());
        // initialize tracing
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .compact()
            .with_line_number(true)
            .init();

        Some((
            // Map response body is required as trace layer changes response body if used within an optional layer
            // https://github.com/tokio-rs/axum/discussions/3439
            tower_http::map_response_body::MapResponseBodyLayer::new(axum::body::Body::new),
            TraceLayer::new_for_http()
                .on_request(trace::DefaultOnRequest::new().level(Level::INFO))
                .on_response(trace::DefaultOnResponse::new().level(Level::INFO))
                .make_span_with(trace::DefaultMakeSpan::new().level(Level::INFO)),
        ))
    } else {
        None
    };

    let model_cache: Cache<String, Arc<Mutex<Session>>> = Cache::new(MAX_CACHED_MODELS);
    let input_cache: Cache<String, Arc<Mutex<CachedInput>>> = Cache::new(MAX_CACHED_MODELS);
    let state = AppState {
        model_cache,
        input_cache,
    };

    // build our application with a route
    let app = Router::new()
        .route("/", post(handle_request))
        .layer(
            service_builder
                .option_layer(trace_layer)
                .layer(DefaultBodyLimit::max(1024 * 1024 * 100)),
        )
        .with_state(state);

    println!("Serving on port {}", config.port);
    let listener = tokio::net::TcpListener::bind(format!(":::{}", config.port))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
struct PredictionRequest {
    /// If true: fetch input from `input_resource` (HTTP GET), cache it, run inference, update cached input.
    /// If false: `input` contains new covariate values; update cached covariates, run inference, update cached sensors.
    init: bool,

    /// Flattened array JSON string (Vec<f32>), used only when init=false (new covariate values).
    /// Example for one covariate: "[1.0]"
    input: Option<String>,

    model_url: String,

    /// Required when init=true. URL to PHP script returning JSON containing key "sensor_".
    input_resource: Option<String>,

    /// Number of sensors (output is [1, sensors]).
    sensors: usize,

    /// Number of covariates appended after all sensor histories.
    covariates: usize,

    /// Length of each sensor/covariate history window (oldest -> newest).
    input_size: usize,
}

fn log_available_providers() {
    println!("CUDA available: {}", CUDAExecutionProvider::default().is_available().unwrap_or(false));
    println!("ROCm available: {}", ROCmExecutionProvider::default().is_available().unwrap_or(false));
}

#[axum::debug_handler]
async fn handle_request(State(state): State<AppState>, Form(request): Form<PredictionRequest>) -> Response {
    let model_url = request.model_url.as_str();

    // 1) Load (and cache) the ONNX model
    let model = state.model_cache.get(model_url).await;
    let local_model = match model {
        Some(model) => model.clone(),
        None => {
            let client = reqwest::Client::new();
            let res = client.get(model_url).send().await;
            let res = match res {
                Ok(response) => match response.error_for_status() {
                    Ok(res) => match res.bytes().await {
                        Ok(model_file) => construct_model(&model_file, GraphOptimizationLevel::Level3, 1),
                        Err(err) => {
                            println!("Encountered error retrieving model: {:?}", err);
                            Err(err.into())
                        }
                    },
                    Err(err) => Err(err.into()),
                },
                Err(err) => Err(err.into()),
            };

            match res {
                Ok(model) => {
                    let model = Arc::new(Mutex::new(model));
                    state
                        .model_cache
                        .insert(model_url.to_owned(), model.clone())
                        .await;
                    model
                }
                Err(err) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("{:?}", err)).into_response();
                }
            }
        }
    };

    // 2) Derive input shape from the ONNX model metadata
    let shape: Vec<usize> = {
        let model_lock = local_model.lock().await;
        let input_info = match model_lock.inputs.first() {
            Some(i) => i,
            None => return (StatusCode::INTERNAL_SERVER_ERROR, "Model has no inputs".to_string()).into_response(),
        };
        match &input_info.input_type {
            ValueType::Tensor { ty: TensorElementType::Float32, shape: tensor_shape, .. } => {
                tensor_shape.iter().map(|&d| {
                    if d < 0 {
                        // Dynamic dimension — treat as 1 (batch)
                        1usize
                    } else {
                        d as usize
                    }
                }).collect()
            }
            other => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Unexpected model input type: {:?}", other),
                ).into_response();
            }
        }
    };
    let shape_elems: usize = shape.iter().product();

    let expected_total = (request.sensors + request.covariates) * request.input_size;
    if shape_elems != expected_total {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "Model input element count ({}) does not match expected (sensors+covariates)*input_size = ({}+{})*{} = {}",
                shape_elems,
                request.sensors,
                request.covariates,
                request.input_size,
                expected_total
            ),
        )
            .into_response();
    }

    // 3) Acquire or (re)initialize cached input
    if request.init {
        let input_resource = match request.input_resource.as_deref() {
            Some(u) if !u.is_empty() => u,
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    "init=true requires input_resource".to_string(),
                )
                    .into_response();
            }
        };

        // Fetch JSON via HTTP GET
        let client = reqwest::Client::new();
        let res = match client.get(input_resource).send().await {
            Ok(r) => match r.error_for_status() {
                Ok(ok) => ok,
                Err(err) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("input_resource returned error status: {:?}", err),
                    )
                        .into_response();
                }
            },
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to GET input_resource: {:?}", err),
                )
                    .into_response();
            }
        };

        let json: serde_json::Value = match res.json().await {
            Ok(v) => v,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to parse JSON from input_resource: {:?}", err),
                )
                    .into_response();
            }
        };

        // Expect: { "sensor_": [[...]] , ... }
        let sensor_val = match json.get("sensor_") {
            Some(v) => v,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    r#"input_resource JSON missing key "sensor_""#.to_string(),
                )
                    .into_response();
            }
        };

        let mut flattened: Vec<f32> = Vec::with_capacity(expected_total);
        match sensor_val {
            serde_json::Value::Array(rows) => {
                for row in rows {
                    match row {
                        serde_json::Value::Array(cols) => {
                            for col in cols {
                                match col.as_f64() {
                                    Some(x) => flattened.push(x as f32),
                                    None => {
                                        return (
                                            StatusCode::BAD_REQUEST,
                                            "sensor_ contains non-numeric value".to_string(),
                                        )
                                            .into_response();
                                    }
                                }
                            }
                        }
                        _ => {
                            return (
                                StatusCode::BAD_REQUEST,
                                "sensor_ must be an array of arrays".to_string(),
                            )
                                .into_response();
                        }
                    }
                }
            }
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    "sensor_ must be an array".to_string(),
                )
                    .into_response();
            }
        }

        if flattened.len() != expected_total {
            return (
                StatusCode::BAD_REQUEST,
                format!(
                    r#"Flattened sensor_ length ({}) does not match expected_total ({})"#,
                    flattened.len(),
                    expected_total
                ),
            )
                .into_response();
        }

        let cached = CachedInput {
            data: flattened,
            sensors: request.sensors,
            covariates: request.covariates,
            input_size: request.input_size,
        };

        state
            .input_cache
            .insert(model_url.to_owned(), Arc::new(Mutex::new(cached)))
            .await;
    } else {
        // init=false requires existing cache
        if state.input_cache.get(model_url).await.is_none() {
            return (
                StatusCode::BAD_REQUEST,
                "init=false received but no cached input exists; call init=true first".to_string(),
            )
                .into_response();
        }
    }

    // 4) Update covariates if init=false (then we always run inference)
    if !request.init {
        let cov_values_json = match request.input.as_deref() {
            Some(s) => s,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    "init=false requires input (new covariate values)".to_string(),
                )
                    .into_response();
            }
        };

        let cov_values: Vec<f32> = match serde_json::from_str(cov_values_json) {
            Ok(v) => v,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid input JSON for covariates: {:?}", err),
                )
                    .into_response();
            }
        };

        if cov_values.len() != request.covariates {
            return (
                StatusCode::BAD_REQUEST,
                format!(
                    "Expected {} covariate values, got {}",
                    request.covariates,
                    cov_values.len()
                ),
            )
                .into_response();
        }

        let cached_arc = state.input_cache.get(model_url).await.unwrap();
        let mut cached = cached_arc.lock().await;

        // If a new init=true happened with different params, reject mismatched updates
        if cached.sensors != request.sensors
            || cached.covariates != request.covariates
            || cached.input_size != request.input_size
        {
            return (
                StatusCode::BAD_REQUEST,
                "Cached input dimensions do not match request; call init=true again".to_string(),
            )
                .into_response();
        }

        // Cached layout:
        // [ sensors * input_size ][ covariates * input_size ]
        let cov_base = cached.sensors * cached.input_size;

        for c in 0..cached.covariates {
            let start = cov_base + c * cached.input_size;
            let end = start + cached.input_size;
            shift_append(&mut cached.data[start..end], cov_values[c]);
        }
    }

    // 5) Run inference on current cached input
    let cached_arc = match state.input_cache.get(model_url).await {
        Some(v) => v,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error: cached input missing after init handling".to_string(),
            )
                .into_response();
        }
    };

    let mut cached = cached_arc.lock().await;

    // The cache is stored "flat by block":
    // [ sensors * input_size ][ covariates * input_size ]
    //
    // Some models expect an interleaved last-dimension layout, e.g. shape:
    //   [1, input_size, sensors+covariates]
    // representing:
    //   input[0][t] = [s0_t, s1_t, ..., sN_t, c0_t, c1_t, ..., cM_t]
    //
    // If the request shape is exactly [1, expected_total], we keep the flat-by-block data.
    // If the request shape is [1, input_size, sensors+covariates], we build interleaved-by-time data.
    let input_data: Vec<f32> = if shape.as_slice() == [1usize, expected_total] {
        cached.data.clone()
    } else if shape.len() == 3
        && shape[0] == 1
        && shape[1] == cached.input_size
        && shape[2] == (cached.sensors + cached.covariates)
    {
        let sensors_base = 0usize;
        let cov_base = cached.sensors * cached.input_size;

        let mut interleaved = Vec::with_capacity(expected_total);

        for t in 0..cached.input_size {
            // sensors first
            for s in 0..cached.sensors {
                let idx = sensors_base + s * cached.input_size + t;
                interleaved.push(cached.data[idx]);
            }
            // then covariates (your "modes", e.g. 17)
            for c in 0..cached.covariates {
                let idx = cov_base + c * cached.input_size + t;
                interleaved.push(cached.data[idx]);
            }
        }

        interleaved
    } else {
        // Keep existing behavior for other shapes (Tensor::from_array will validate element count)
        cached.data.clone()
    };

    let input_tensor = match Tensor::from_array((shape, input_data)) {
        Ok(input) => input,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };

    let mut model_lock = local_model.lock().await;
    let out = match model_lock.run(ort::inputs![input_tensor]) {
        Ok(out) => out,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };

    // Try common output names first, then fall back to first output
    let res = if let Some(a) = out.get("variable") {
        a
    } else if let Some(a) = out.get("values") {
        a
    } else if let Some(a) = out.get("output") {
        a
    } else {
        // If none of the common names work, just use the first output
        &match out.iter().next() {
            Some((name, value)) => {
                println!("Using first output with name: {}", name);
                value
            }
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Model produced no outputs".to_string()
                ).into_response()
            }
        }
    };

    let pred_arr = match res.try_extract_array::<f32>() {
        Ok(a) => a,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response()
    };

    let preds: Vec<f32> = pred_arr.iter().copied().collect();
    // Determine output format and extract forecast values
    let forecast: Vec<f32> = if preds.len() == cached.sensors {
        // Case 1: Output shape is [n_sensors] - direct prediction (backwards compatible)
        preds
    } else if preds.len() == cached.input_size * cached.sensors {
        // Case 2: Output shape is (1, input_size, n_sensors, 1) or (input_size, n_sensors)
        // Extract prediction[0, -1, :, 0] (last timestep for all sensors)
        let last_timestep_idx = cached.input_size - 1;
        (0..cached.sensors)
            .map(|s| {
                let idx = last_timestep_idx * cached.sensors + s;
                preds[idx]
            })
            .collect()
    } else {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "Model output length ({}) does not match expected formats: [{}] or [{} * {}]",
                preds.len(),
                cached.sensors,
                cached.input_size,
                cached.sensors
            ),
        )
            .into_response();
    };

    // 6) Update sensor histories with predicted values (shift + append)
    for s in 0..cached.sensors {
        let start = s * cached.input_size;
        let end = start + cached.input_size;
        shift_append(&mut cached.data[start..end], forecast[s]);
    }

    let body = match serde_json::to_string(&forecast) {
        Ok(s) => s,
        Err(err) => format!("{:?}", err),
    };
    (StatusCode::OK, body).into_response()
}

fn shift_append(slice: &mut [f32], new_val: f32) {
    if slice.is_empty() {
        return;
    }
    // shift left by 1 (drop oldest)
    for i in 0..(slice.len() - 1) {
        slice[i] = slice[i + 1];
    }
    // append newest
    let last = slice.len() - 1;
    slice[last] = new_val;
}

fn construct_model(model_bytes: &[u8], level: GraphOptimizationLevel, threads: usize) -> anyhow::Result<Model> {
    let model = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),   // NVIDIA GPU
            ROCmExecutionProvider::default().build(),   // AMD GPU
            CPUExecutionProvider::default().build(),    // Fallback
        ])?
        .with_optimization_level(level)?
        .with_intra_threads(threads)?
        .commit_from_memory(model_bytes)?;
    Ok(model)
}
// ... existing code ...

#[cfg(test)]
mod test {
    use std::{fs::read, time::Instant};

    use ort::{session::builder::GraphOptimizationLevel, value::Tensor};

    use crate::construct_model;

    #[test]
    fn test_model_creation_d() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let level = GraphOptimizationLevel::Disable;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l1() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let level = GraphOptimizationLevel::Level1;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l2() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let level = GraphOptimizationLevel::Level2;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l3() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let level = GraphOptimizationLevel::Level3;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_inference() {
        let data = vec![
            0.820045_f32,
            0.820045,
            0.69701624,
            0.69701624,
            0.6086147,
            0.6086147,
            0.51213145,
            0.44877553,
            0.44877553,
            0.44877553,
            0.33946013,
            0.2616396,
        ];
        let input = Tensor::from_array(([1usize, 12], data)).unwrap();
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let mut model = construct_model(&model_file, GraphOptimizationLevel::Level3, 1).unwrap();
        let start = Instant::now();
        let res = model.run(ort::inputs![input]).unwrap();
        let elapsed = start.elapsed().as_micros();
        println!("Elapsed time: {}", elapsed);
        let res = res["variable"].try_extract_array::<f32>().unwrap()[[0, 0]];
        println!("Res: {:?}", res)
    }
}