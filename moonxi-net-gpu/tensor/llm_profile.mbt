// SPDX-FileCopyrightText: 2026 chnlkw
// SPDX-License-Identifier: MIT

///|
/// Optional profiling state for native LLM operations.
pub struct LlmProfiler {
  mut enabled : Bool
  mut counts : Map[String, Int64]
  mut host_ms : Map[String, Double]
  mut cuda_ms : Map[String, Float]
}

///|
let global_llm_profiler : LlmProfiler = {
  enabled: false,
  counts: {},
  host_ms: {},
  cuda_ms: {},
}

///|
pub fn llm_profiler() -> LlmProfiler {
  global_llm_profiler
}

///|
pub fn LlmProfiler::reset(self : LlmProfiler, enabled~ : Bool) -> Unit {
  self.enabled = enabled
  self.counts = {}
  self.host_ms = {}
  self.cuda_ms = {}
}

///|
pub fn LlmProfiler::count(
  self : LlmProfiler,
  name : String,
  amount : Int64,
) -> Unit {
  if self.enabled {
    self.counts[name] = self.counts.get(name).unwrap_or(0L) + amount
  }
}

///|
fn LlmProfiler::add_host(
  self : LlmProfiler,
  name : String,
  started_at : Double,
) -> Unit {
  if self.enabled {
    let elapsed = @cuda.get_time_ms() - started_at
    self.host_ms[name] = self.host_ms.get(name).unwrap_or(0.0) + elapsed
  }
}

///|
pub fn[T] LlmProfiler::host(
  self : LlmProfiler,
  name : String,
  operation : () -> T,
) -> T {
  if !self.enabled {
    return operation()
  }
  let started_at = @cuda.get_time_ms()
  let result = operation()
  self.add_host(name, started_at)
  result
}

///|
pub fn LlmProfiler::cuda(
  self : LlmProfiler,
  name : String,
  operation : () -> Unit,
) -> Unit {
  if !self.enabled {
    return operation()
  }
  let start = @cuda.event_create()
  ignore(@cuda.event_record(@cuda.null_buffer(), start))
  operation()
  let end = @cuda.event_create()
  ignore(@cuda.event_record(@cuda.null_buffer(), end))
  ignore(@cuda.event_synchronize(end))
  let elapsed = @cuda.event_elapsed_ms(start, end)
  self.cuda_ms[name] = self.cuda_ms.get(name).unwrap_or(0.0F) + elapsed
}

///|
pub fn LlmProfiler::print(self : LlmProfiler) -> Unit {
  let output = StringBuilder()
  output.write_string("\n=== Native LLM profile ===")
  let sections = [
    ("counts", self.counts.keys().collect()),
    ("host_ms", self.host_ms.keys().collect()),
    ("cuda_ms", self.cuda_ms.keys().collect()),
  ]
  for section in sections {
    let names = section.1
    names.sort()
    for name in names {
      output.write_string("\n\{section.0} \{name}: ")
      match section.0 {
        "counts" => output.write_string(self.counts[name].to_string())
        "host_ms" => output.write_string(self.host_ms[name].to_string())
        "cuda_ms" => output.write_string(self.cuda_ms[name].to_string())
        _ => ()
      }
    }
  }
  if self.enabled {
    output.write_string("\nworkspace_peak_bytes: \{gpu_pool_peak()}")
    output.write_string("\nworkspace_used_bytes: \{gpu_pool_used()}")
    output.write_string(
      "\nCUDA event timing synchronizes each measured operation.",
    )
  }
  println(output.to_string())
}
