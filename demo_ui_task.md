# Live-demo UI for the real-model streaming engine (OpenCode task)

Location: /home/bhandi/opencode-test

Goal: add a browser UI that shows the REAL MODEL streaming live. Build these TWO
files:

1) web/demo_server.py
   - Starts a RangeServer on port 8001 serving models/tinyllama.safetensors with
     a throttled link (bandwidth_mbps=1000.0).
   - Runs the REAL forward pass in a background thread using
     RealModelComputeEngine (auto device: cuda when available).
   - Feeds layers one at a time: find each layer payload via the RangeServer
     (range fetch), run REAL.run_layer(name, payload), then append a per-layer
     result {layer, download_s, compute_s, checksum, timestamp} to
     web/live_results.json IMMEDIATELY (so the UI is live, not at the end).
   - Writes a final summary to web/results.json (first_layer_s, total_s,
     compute_s, io_wait_s, masking_ratio, peak_rss_mb).
   - Serves on port 8000: /demo.html and /results.json (plus proxy the model
     route to port 8001).
   - NOTE: the demo engine must also accept `engine.run_layer(name, payload)`
     which it does (RealModelComputeEngine has that). It carries `self.hidden`
     across layers. Use the SAME loop as the benchmark.

2) web/demo.html
   - Browser UI: button "Run demo", config for bandwidth, and a live scrolling
     table of per-layer {layer, dl_s, compute_s, checksum}.
   - A line chart (canvas) of checksum rising as layers compute.
   - Shows the final summary (first_layer_s, total_s, compute_s, masking_ratio).
   - Uses only std JS (fetch /results.json, canvas). No external libs.

Run instructions to include in the file:
  cd /home/bhandi/opencode-test
  python3 web/demo_server.py
  # then open http://127.0.0.1:8000/demo.html

Important: do NOT fabricate numbers. The checksum must come from
REAL.run_layer() (real BF16 weights). Run on CPU (torch CPU, seq=1) is fine —
state the runtime in the UI ("torch 2.15.0+cpu / CUDA: available").

Verify after writing:
  python3 -m py_compile web/demo_server.py  (must pass)
  open the HTML with no JS syntax errors (long ids, no undefined vars).
