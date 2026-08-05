fn main() {
    #[cfg(feature = "mobile")]
    {
        uniffi_build::generate_scaffolding("./src/kchat.udl").unwrap();
    }

    #[cfg(feature = "desktop")]
    {
        napi_build::setup();
    }
}
