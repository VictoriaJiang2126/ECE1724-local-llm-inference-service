#[macro_use]
extern crate rocket;

mod api;
mod model_registry;
mod types;

use std::sync::Arc;

use api::{health, infer, list_models, load_model};
use model_registry::ModelRegistry;

#[launch]
fn rocket() -> _ {
    let registry = Arc::new(ModelRegistry::new());

    rocket::build()
        .manage(registry)
        .mount(
            "/",
            routes![
                health,
                list_models,
                load_model,
                infer,
            ],
        )
}
