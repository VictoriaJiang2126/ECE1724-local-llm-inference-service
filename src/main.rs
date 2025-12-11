#[macro_use]
extern crate rocket;

mod api;
mod app_state;
mod engine;
mod model_registry;
mod types;

use std::sync::Arc;

use api::{health, infer, infer_stream, infer_stream_get, list_models, load_model};
use app_state::AppState;





#[launch]
fn rocket() -> _ {
    let max_concurrent_infer = 10;
    let state = AppState::new(max_concurrent_infer);

    rocket::build()
        .manage(state as Arc<AppState>)
        .mount(
            "/",
            routes![
                health,
                list_models,
                load_model,
                infer,              // POST /infer         （非流式）
                infer_stream,       // POST /infer?stream=true （curl 用）
                infer_stream_get,   // GET  /infer_stream?model_name=&prompt= （前端用）
            ],
        )
        .mount("/", rocket::fs::FileServer::from("static"))
}
