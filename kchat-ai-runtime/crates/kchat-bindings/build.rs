fn main() {
    // UniFFI proc-macro mode (#[uniffi::export] + setup_scaffolding!) needs no
    // build-time scaffolding generation — the macros emit it inline.

    #[cfg(feature = "desktop")]
    {
        napi_build::setup();
    }
}
