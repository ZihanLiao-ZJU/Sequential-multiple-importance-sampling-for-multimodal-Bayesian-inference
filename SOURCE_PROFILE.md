# Registered settings

{
  "profile": "author_latest with explicit coordinate-pi ES each_step overrides",
  "estimator": "source_legacy_mis",
  "sampler": {
    "name": "ES",
    "ESMode": "coordinate",
    "AcceptArithmetic": "exp",
    "SeedUpdateMode": "each_step",
    "Ntry": 20,
    "SedSav": false
  }
}

Only the explicit run_qualified entrypoint selects these registered settings. Constructor defaults and legacy drivers retain their previous behavior.
