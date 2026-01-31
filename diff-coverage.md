# Diff Coverage

## Diff: origin/main...HEAD, staged and unstaged changes

- src/oscura/**init**&#46;py (0.0%): Missing lines 56
- src/oscura/analyzers/eye/**init**&#46;py (100%)
- src/oscura/analyzers/eye/generation&#46;py (19.1%): Missing lines 97-98,100-102,104-105,108-109,111,115,130-132,135,140-141,155-157,159-160,162,164,166-167,174,185-187,189-191,193-194,196-197,199-200,207,218-219,221-222,224-225,227,234,267-268,270-271,279-280,283,286-288,290,292-293,295-296,299-301,303-304,307-308,310,312-313,315-316,323-324,326,346-350,363-364,366-368,370-371,373,389-390,392-394,396-398,400-401,403,405,407,419-422,456-457,460-462,464-465,467,472,475-477,480-481,483
- src/oscura/analyzers/jitter/**init**&#46;py (100%)
- src/oscura/analyzers/jitter/timing&#46;py (83.5%): Missing lines 101-102,105,108-109,112-114,117,119,152,163,184-185,226,236,308,358,388,403
- src/oscura/analyzers/patterns/**init**&#46;py (100%)
- src/oscura/analyzers/patterns/reverse_engineering&#46;py (87.9%): Missing lines 239,243,250-251,266-267,273-274,287-288,304-305,485,559,563,572,592,596,627,665,678,680,689,705,710,756,921-923,926-927,931,964-966,969-970,973
- src/oscura/analyzers/statistics/basic&#46;py (58.8%): Missing lines 358,362,380,383,386,389-390,393-398,400
- src/oscura/analyzers/statistics/correlation&#46;py (100%)
- src/oscura/analyzers/waveform/measurements&#46;py (90.2%): Missing lines 372,428,446,940
- src/oscura/analyzers/waveform/spectral&#46;py (63.5%): Missing lines 708,2069,2083,2088,2239,2248-2249,2251,2253,2255-2259,2261-2262,2264,2267-2272,2274-2275,2277-2280,2282,2284
- src/oscura/automotive/**init**&#46;py (0.0%): Missing lines 52
- src/oscura/core/types&#46;py (100%)
- src/oscura/reporting/**init**&#46;py (100%)
- src/oscura/reporting/automation&#46;py (91.4%): Missing lines 145,165,260,263-264,340-341,343-344
- src/oscura/reporting/citations&#46;py (97.6%): Missing lines 254,372
- src/oscura/reporting/core&#46;py (100%)
- src/oscura/reporting/formatting/**init**&#46;py (100%)
- src/oscura/reporting/formatting/measurements&#46;py (67.6%): Missing lines 44,76-77,81-82,118,121,128-130,132-135,137,140-141,176,181,198,218-219
- src/oscura/reporting/html&#46;py (82.4%): Missing lines 553,680,695
- src/oscura/reporting/interpretation&#46;py (89.1%): Missing lines 121-122,127-128,143-145,161-162,184-185,211-213,215-217,219-220,265
- src/oscura/reporting/summary&#46;py (43.2%): Missing lines 53,56-57,60-61,64-67,70,72-74,76-78,80-83,85,93,95-98,101-103,105-113,115-116,118,127-129,131,145,147,159,161-162,165,167,249,258,265,267,273,276-283,321,326
- src/oscura/reporting/visualization&#46;py (18.9%): Missing lines 73-74,77-82,85-86,89-91,94,97,122-123,125-127,156,159,162-165,174-175,200,203,206,208-210,212-218,227,230-231,233,258,261,263-264,267-268,270,295,298,301,306-307,309,311,335,338-339,341-346,354-355,378,381,386-389,395,398-399,410-412,432-433,436-437,446,449-451,454-456,458,461,472,474,497,500-501,504-505,508-511,513-514,516,535-540,542
- src/oscura/visualization/**init**&#46;py (100%)
- src/oscura/visualization/batch&#46;py (75.7%): Missing lines 99-101,332-334,392-395,398-401,403-406,408-410,443,454-457,465-468,473-476,481-484,488-489,492,501-504
- src/oscura/workflows/**init**&#46;py (100%)
- src/oscura/workflows/waveform&#46;py (4.2%): Missing lines 125-126,128-129,132-134,136-139,141-145,148,151,155-160,163,165-169,172-173,176-179,181-185,187,189,191,194-199,201-202,204,209,211-215,218-219,222-223,225-226,228,230,232-233,235,238,240,243-244,246,249-252,254-256,259-262,264-268,270,272,274-278,281-282,284-288,290-291,294,296-297,299-300,302,308-309,316-317,319-320,324,327,329,331-332,334-336,339,341,347-351,356,358-359,362-365,367,369-374,376-377,379,387,400-401,404-412,414-421,424-425,427-431,433,436-437,440-443,445-446,452,462,464-473,476-478,481,483-484,494-498,501-506,508,511-512,514-515,518-523,525,528,531,537,549,553,555-556,558-559,561,564-565,571-572,575-576,582-583,589-590,596-597,603,606-607,609-611,614-615,617-618,620-630,633,659-668,670-671,673,685,687-688,690-692,694-695,697-698,700-701,703-704,706-707,709-713,715,717-721,723,735-736,739-742,744-750,752,764,766-776,778

## Summary

- **Total**: 2073 lines
- **Missing**: 856 lines
- **Coverage**: 58%


## src/oscura/**init**&#46;py

Lines 52-60

```python
  52 
  53     __version__ = version("oscura")
  54 except Exception:
  55     # Fallback for development/testing when package not installed
! 56     __version__ = "0.8.0"
  57 
  58 __author__ = "Oscura Contributors"
  59 
  60 # Core types
```


---


## src/oscura/analyzers/eye/generation&#46;py

Lines 93-119

```python
   93 
   94     References:
   95         OIF CEI: Common Electrical I/O Eye Diagram Methodology
   96     """
!  97     data = trace.data
!  98     sample_rate = trace.metadata.sample_rate
   99 
! 100     samples_per_ui = _validate_unit_interval(unit_interval, sample_rate)
! 101     total_ui_samples = samples_per_ui * n_ui
! 102     _validate_data_length(len(data), total_ui_samples)
  103 
! 104     trigger_indices = _find_trigger_points(data, trigger_level, trigger_edge)
! 105     eye_traces = _extract_eye_traces(
  106         data, trigger_indices, samples_per_ui, total_ui_samples, max_traces
  107     )
! 108     eye_data = np.array(eye_traces, dtype=np.float64)
! 109     time_axis = np.linspace(0, n_ui, total_ui_samples, endpoint=False)
  110 
! 111     histogram, voltage_bins, time_bins = _generate_histogram_if_requested(
  112         eye_data, time_axis, n_ui, generate_histogram, histogram_bins
  113     )
  114 
! 115     return EyeDiagram(
  116         data=eye_data,
  117         time_axis=time_axis,
  118         unit_interval=unit_interval,
  119         samples_per_ui=samples_per_ui,
```


---


Lines 126-145

```python
  126 
  127 
  128 def _validate_unit_interval(unit_interval: float, sample_rate: float) -> int:
  129     """Validate unit interval and calculate samples per UI."""
! 130     samples_per_ui = round(unit_interval * sample_rate)
! 131     if samples_per_ui < 4:
! 132         raise AnalysisError(
  133             f"Unit interval too short: {samples_per_ui} samples/UI. Need at least 4 samples per UI."
  134         )
! 135     return samples_per_ui
  136 
  137 
  138 def _validate_data_length(n_samples: int, total_ui_samples: int) -> None:
  139     """Validate that we have enough data for eye generation."""
! 140     if n_samples < total_ui_samples * 2:
! 141         raise InsufficientDataError(
  142             f"Need at least {total_ui_samples * 2} samples for eye diagram",
  143             required=total_ui_samples * 2,
  144             available=n_samples,
  145             analysis_type="eye_diagram_generation",
```


---


Lines 151-171

```python
  151     trigger_level: float,
  152     trigger_edge: str,
  153 ) -> NDArray[np.intp]:
  154     """Find trigger points in the data."""
! 155     low = np.percentile(data, 10)
! 156     high = np.percentile(data, 90)
! 157     threshold = low + trigger_level * (high - low)
  158 
! 159     if trigger_edge == "rising":
! 160         trigger_mask = (data[:-1] < threshold) & (data[1:] >= threshold)
  161     else:
! 162         trigger_mask = (data[:-1] >= threshold) & (data[1:] < threshold)
  163 
! 164     trigger_indices = np.where(trigger_mask)[0]
  165 
! 166     if len(trigger_indices) < 2:
! 167         raise InsufficientDataError(
  168             "Not enough trigger events for eye diagram",
  169             required=2,
  170             available=len(trigger_indices),
  171             analysis_type="eye_diagram_generation",
```


---


Lines 170-178

```python
  170             available=len(trigger_indices),
  171             analysis_type="eye_diagram_generation",
  172         )
  173 
! 174     return trigger_indices
  175 
  176 
  177 def _extract_eye_traces(
  178     data: NDArray[np.float64],
```


---


Lines 181-204

```python
  181     total_ui_samples: int,
  182     max_traces: int | None,
  183 ) -> list[NDArray[np.float64]]:
  184     """Extract eye traces from data using trigger points."""
! 185     eye_traces = []
! 186     half_ui = samples_per_ui // 2
! 187     n_samples = len(data)
  188 
! 189     for trig_idx in trigger_indices:
! 190         start_idx = trig_idx - half_ui
! 191         end_idx = start_idx + total_ui_samples
  192 
! 193         if start_idx >= 0 and end_idx <= n_samples:
! 194             eye_traces.append(data[start_idx:end_idx])
  195 
! 196         if max_traces is not None and len(eye_traces) >= max_traces:
! 197             break
  198 
! 199     if len(eye_traces) == 0:
! 200         raise InsufficientDataError(
  201             "Could not extract any complete eye traces",
  202             required=1,
  203             available=0,
  204             analysis_type="eye_diagram_generation",
```


---


Lines 203-211

```python
  203             available=0,
  204             analysis_type="eye_diagram_generation",
  205         )
  206 
! 207     return eye_traces
  208 
  209 
  210 def _generate_histogram_if_requested(
  211     eye_data: NDArray[np.float64],
```


---


Lines 214-231

```python
  214     generate_histogram: bool,
  215     histogram_bins: tuple[int, int],
  216 ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
  217     """Generate 2D histogram if requested."""
! 218     if not generate_histogram:
! 219         return None, None, None
  220 
! 221     all_voltages = eye_data.flatten()
! 222     all_times = np.tile(time_axis, len(eye_data))
  223 
! 224     voltage_range = (np.min(all_voltages), np.max(all_voltages))
! 225     time_range = (0, n_ui)
  226 
! 227     histogram, voltage_edges, time_edges = np.histogram2d(
  228         all_voltages,
  229         all_times,
  230         bins=histogram_bins,
  231         range=[voltage_range, time_range],
```


---


Lines 230-238

```python
  230         bins=histogram_bins,
  231         range=[voltage_range, time_range],
  232     )
  233 
! 234     return histogram, voltage_edges, time_edges
  235 
  236 
  237 def generate_eye_from_edges(
  238     trace: WaveformTrace,
```


---


Lines 263-275

```python
  263     Example:
  264         >>> edges = recover_clock_edges(trace)
  265         >>> eye = generate_eye_from_edges(trace, edges)
  266     """
! 267     data = trace.data
! 268     sample_rate = trace.metadata.sample_rate
  269 
! 270     if len(edge_timestamps) < 3:
! 271         raise InsufficientDataError(
  272             "Need at least 3 edge timestamps",
  273             required=3,
  274             available=len(edge_timestamps),
  275             analysis_type="eye_diagram_generation",
```


---


Lines 275-320

```python
  275             analysis_type="eye_diagram_generation",
  276         )
  277 
  278     # Calculate unit interval from edges
! 279     periods = np.diff(edge_timestamps)
! 280     unit_interval = float(np.median(periods))
  281 
  282     # Create time vector for original data
! 283     original_time = np.arange(len(data)) / sample_rate
  284 
  285     # Extract and resample traces around each edge
! 286     eye_traces = []
! 287     total_samples = samples_per_ui * n_ui
! 288     half_ui = unit_interval / 2
  289 
! 290     for edge_time in edge_timestamps:
  291         # Define window around edge
! 292         start_time = edge_time - half_ui
! 293         end_time = start_time + unit_interval * n_ui
  294 
! 295         if start_time < 0 or end_time > original_time[-1]:
! 296             continue
  297 
  298         # Find samples within window
! 299         mask = (original_time >= start_time) & (original_time <= end_time)
! 300         window_time = original_time[mask] - start_time
! 301         window_data = data[mask]
  302 
! 303         if len(window_data) < 4:
! 304             continue
  305 
  306         # Resample to consistent samples_per_ui
! 307         resample_time = np.linspace(0, unit_interval * n_ui, total_samples)
! 308         resampled = np.interp(resample_time, window_time, window_data)
  309 
! 310         eye_traces.append(resampled)
  311 
! 312         if max_traces is not None and len(eye_traces) >= max_traces:
! 313             break
  314 
! 315     if len(eye_traces) == 0:
! 316         raise InsufficientDataError(
  317             "Could not extract any eye traces",
  318             required=1,
  319             available=0,
  320             analysis_type="eye_diagram_generation",
```


---


Lines 319-330

```python
  319             available=0,
  320             analysis_type="eye_diagram_generation",
  321         )
  322 
! 323     eye_data = np.array(eye_traces, dtype=np.float64)
! 324     time_axis = np.linspace(0, n_ui, total_samples, endpoint=False)
  325 
! 326     return EyeDiagram(
  327         data=eye_data,
  328         time_axis=time_axis,
  329         unit_interval=unit_interval,
  330         samples_per_ui=samples_per_ui,
```


---


Lines 342-354

```python
  342 
  343     Returns:
  344         Threshold value.
  345     """
! 346     low = np.percentile(data, 10)
! 347     high = np.percentile(data, 90)
! 348     amplitude_range = high - low
! 349     threshold: float = float(low + trigger_fraction * amplitude_range)
! 350     return threshold
  351 
  352 
  353 def _find_trace_crossings(data: NDArray[np.float64], threshold: float) -> list[int]:
  354     """Find crossing indices for all traces.
```


---


Lines 359-377

```python
  359 
  360     Returns:
  361         List of crossing indices for traces with crossings.
  362     """
! 363     n_traces, _samples_per_trace = data.shape
! 364     crossing_indices = []
  365 
! 366     for trace_idx in range(n_traces):
! 367         trace = data[trace_idx, :]
! 368         crossings = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]
  369 
! 370         if len(crossings) > 0:
! 371             crossing_indices.append(crossings[0])
  372 
! 373     return crossing_indices
  374 
  375 
  376 def _align_traces_to_target(
  377     data: NDArray[np.float64], threshold: float, target_crossing: int
```


---


Lines 385-411

```python
  385 
  386     Returns:
  387         Aligned trace data.
  388     """
! 389     n_traces, _samples_per_trace = data.shape
! 390     aligned_data = np.zeros_like(data)
  391 
! 392     for trace_idx in range(n_traces):
! 393         trace = data[trace_idx, :]
! 394         crossings = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]
  395 
! 396         if len(crossings) > 0:
! 397             crossing = crossings[0]
! 398             shift = target_crossing - crossing
  399 
! 400             if shift != 0:
! 401                 aligned_data[trace_idx, :] = np.roll(trace, shift)
  402             else:
! 403                 aligned_data[trace_idx, :] = trace
  404         else:
! 405             aligned_data[trace_idx, :] = trace
  406 
! 407     return aligned_data
  408 
  409 
  410 def _apply_symmetric_centering(data: NDArray[np.float64]) -> NDArray[np.float64]:
  411     """Apply symmetric amplitude centering if enabled.
```


---


Lines 415-426

```python
  415 
  416     Returns:
  417         Symmetrically centered data.
  418     """
! 419     max_abs = np.max(np.abs(data))
! 420     if max_abs > 0:
! 421         data = data - np.mean(data)
! 422     return data
  423 
  424 
  425 def auto_center_eye_diagram(
  426     eye: EyeDiagram,
```


---


Lines 452-487

```python
  452 
  453     References:
  454         VIS-021: Eye Diagram Auto-Centering
  455     """
! 456     if not 0 <= trigger_fraction <= 1:
! 457         raise ValueError(f"trigger_fraction must be in [0, 1], got {trigger_fraction}")
  458 
  459     # Setup: calculate threshold and find crossings
! 460     data = eye.data
! 461     threshold = _calculate_trigger_threshold(data, trigger_fraction)
! 462     crossing_indices = _find_trace_crossings(data, threshold)
  463 
! 464     if len(crossing_indices) == 0:
! 465         import warnings
  466 
! 467         warnings.warn(
  468             "No crossing points found, cannot auto-center eye diagram",
  469             UserWarning,
  470             stacklevel=2,
  471         )
! 472         return eye
  473 
  474     # Processing: align traces to target crossing point
! 475     _n_traces, samples_per_trace = data.shape
! 476     target_crossing = samples_per_trace // 2
! 477     aligned_data = _align_traces_to_target(data, threshold, target_crossing)
  478 
  479     # Result building: apply symmetric centering and create result
! 480     if symmetric_range:
! 481         aligned_data = _apply_symmetric_centering(aligned_data)
  482 
! 483     return EyeDiagram(
  484         data=aligned_data,
  485         time_axis=eye.time_axis,
  486         unit_interval=eye.unit_interval,
  487         samples_per_ui=eye.samples_per_ui,
```


---


## src/oscura/analyzers/jitter/timing&#46;py

Lines 97-123

```python
   97 
   98     References:
   99         IEEE 2414-2020 Section 4.2: Time Interval Error Definition
  100     """
! 101     if len(edge_timestamps) < 3:
! 102         return np.array([], dtype=np.float64)
  103 
  104     # Calculate actual periods
! 105     periods = np.diff(edge_timestamps)
  106 
  107     # Use mean period if nominal not provided
! 108     if nominal_period is None:
! 109         nominal_period = np.mean(periods)
  110 
  111     # Calculate ideal edge positions
! 112     n_edges = len(edge_timestamps)
! 113     start_time = edge_timestamps[0]
! 114     ideal_positions = start_time + np.arange(n_edges) * nominal_period
  115 
  116     # TIE is actual - ideal
! 117     tie: NDArray[np.float64] = edge_timestamps - ideal_positions
  118 
! 119     return tie
  120 
  121 
  122 def cycle_to_cycle_jitter(
  123     periods: NDArray[np.float64],
```


---


Lines 148-156

```python
  148     References:
  149         IEEE 2414-2020 Section 5.3: Cycle-to-Cycle Jitter
  150     """
  151     if len(periods) < 3:
! 152         raise InsufficientDataError(
  153             "Cycle-to-cycle jitter requires at least 3 periods",
  154             required=3,
  155             available=len(periods),
  156             analysis_type="cycle_to_cycle_jitter",
```


---


Lines 159-167

```python
  159     # Remove NaN values
  160     valid_periods = periods[~np.isnan(periods)]
  161 
  162     if len(valid_periods) < 3:
! 163         raise InsufficientDataError(
  164             "Cycle-to-cycle jitter requires at least 3 valid periods",
  165             required=3,
  166             available=len(valid_periods),
  167             analysis_type="cycle_to_cycle_jitter",
```


---


Lines 180-189

```python
  180     if include_histogram and len(c2c_values) > 10:
  181         hist, bin_edges = np.histogram(c2c_values, bins=n_bins, density=True)
  182         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  183     else:
! 184         hist = None
! 185         bin_centers = None
  186 
  187     return CycleJitterResult(
  188         c2c_rms=c2c_rms,
  189         c2c_pp=c2c_pp,
```


---


Lines 222-230

```python
  222     References:
  223         IEEE 2414-2020 Section 5.2: Period Jitter
  224     """
  225     if len(periods) < 2:
! 226         raise InsufficientDataError(
  227             "Period jitter requires at least 2 periods",
  228             required=2,
  229             available=len(periods),
  230             analysis_type="period_jitter",
```


---


Lines 232-240

```python
  232 
  233     valid_periods = periods[~np.isnan(periods)]
  234 
  235     if nominal_period is None:
! 236         nominal_period = np.mean(valid_periods)
  237 
  238     # Calculate deviations from nominal
  239     deviations = valid_periods - nominal_period
```


---


Lines 304-312

```python
  304         if len(next_rising) > 0:
  305             low_times.append(next_rising[0] - f_edge)
  306 
  307     if len(high_times) < 1 or len(low_times) < 1:
! 308         raise InsufficientDataError(
  309             "Could not measure high/low times",
  310             required=2,
  311             available=0,
  312             analysis_type="dcd_measurement",
```


---


Lines 354-362

```python
  354     sample_rate = trace.metadata.sample_rate
  355     sample_period = 1.0 / sample_rate
  356 
  357     if len(data) < 3:
! 358         return np.array([]), np.array([])
  359 
  360     # Find amplitude levels - use more extreme percentiles for better accuracy
  361     low = np.percentile(data, 5)
  362     high = np.percentile(data, 95)
```


---


Lines 384-392

```python
  384             frac = max(0.0, min(1.0, frac))
  385             t_offset = frac * sample_period
  386         else:
  387             # Values are equal, use midpoint
! 388             t_offset = sample_period / 2
  389         rising_edges.append(idx * sample_period + t_offset)
  390 
  391     falling_edges = []
  392     for idx in falling_indices:
```


---


Lines 399-407

```python
  399             frac = max(0.0, min(1.0, frac))
  400             t_offset = frac * sample_period
  401         else:
  402             # Values are equal, use midpoint
! 403             t_offset = sample_period / 2
  404         falling_edges.append(idx * sample_period + t_offset)
  405 
  406     return (
  407         np.array(rising_edges, dtype=np.float64),
```


---


## src/oscura/analyzers/patterns/reverse_engineering&#46;py

Lines 235-247

```python
  235         # 2. Classify data type
  236         if entropy_result.encryption_likelihood > 0.7:
  237             data_type = "encrypted"
  238         elif entropy_result.compression_likelihood > 0.7:
! 239             data_type = "compressed"
  240         elif entropy_val < 3.0:
  241             data_type = "structured"
  242         else:
! 243             data_type = "mixed"
  244 
  245         # 3. Signature discovery (skip for encrypted/compressed)
  246         signatures = []
  247         if detect_signatures and data_type in ["structured", "mixed"]:
```


---


Lines 246-255

```python
  246         signatures = []
  247         if detect_signatures and data_type in ["structured", "mixed"]:
  248             try:
  249                 signatures = self.signature_discovery.discover_signatures(data)
! 250             except Exception as e:
! 251                 logger.warning(f"Signature discovery failed: {e}")
  252 
  253         # 4. Repeating pattern detection
  254         repeating = []
  255         try:
```


---


Lines 262-271

```python
  262                     "frequency": seq.frequency,
  263                 }
  264                 for seq in sequences[:10]  # Top 10
  265             ]
! 266         except Exception as e:
! 267             logger.warning(f"Repeating pattern detection failed: {e}")
  268 
  269         # 5. N-gram profiling
  270         ngram_profile = {}
  271         try:
```


---


Lines 269-278

```python
  269         # 5. N-gram profiling
  270         ngram_profile = {}
  271         try:
  272             ngram_profile = self.ngram_analyzer.analyze(data)
! 273         except Exception as e:
! 274             logger.warning(f"N-gram analysis failed: {e}")
  275 
  276         # 6. Anomaly detection (simple z-score based)
  277         anomalies = []
  278         if detect_anomalies:
```


---


Lines 283-292

```python
  283                 std = np.std(byte_array)
  284                 if std > 0:
  285                     z_scores = np.abs((byte_array - mean) / std)
  286                     anomalies = np.where(z_scores > 3.0)[0].tolist()
! 287             except Exception as e:
! 288                 logger.warning(f"Anomaly detection failed: {e}")
  289 
  290         # 7. Periodic pattern detection
  291         periodic = []
  292         try:
```


---


Lines 300-309

```python
  300                         "confidence": period_result.confidence,
  301                         "method": period_result.method,
  302                     }
  303                 )
! 304         except Exception as e:
! 305             logger.warning(f"Period detection failed: {e}")
  306 
  307         # Calculate overall confidence
  308         confidence = self._calculate_analysis_confidence(
  309             entropy_result, signatures, repeating, anomalies
```


---


Lines 481-489

```python
  481             >>> for field in fields:
  482             ...     print(f"Field: {field.field_type} at {field.offset}")
  483         """
  484         if not messages:
! 485             return []
  486 
  487         # Validate all messages same length
  488         msg_len = len(messages[0])
  489         if not all(len(msg) == msg_len for msg in messages):
```


---


Lines 555-567

```python
  555 
  556         if result.encryption_likelihood > 0.7:
  557             return "encrypted"
  558         elif result.compression_likelihood > 0.7:
! 559             return "compressed"
  560         elif result.shannon_entropy < 3.0:
  561             return "structured"
  562         else:
! 563             return "mixed"
  564 
  565     # =========================================================================
  566     # Internal Helper Methods
  567     # =========================================================================
```


---


Lines 568-576

```python
  568 
  569     def _detect_delimiter(self, messages: list[bytes]) -> bytes | None:
  570         """Detect delimiter by finding common endings."""
  571         if len(messages) < 2:
! 572             return None
  573 
  574         # Look for common suffixes (last 1-4 bytes)
  575         for delim_len in range(1, 5):
  576             candidates: dict[bytes, int] = {}
```


---


Lines 588-600

```python
  588 
  589     def _infer_fields(self, messages: list[bytes], min_field_size: int) -> list[FieldDescriptor]:
  590         """Infer field boundaries using entropy and variance analysis."""
  591         if not messages:
! 592             return []
  593 
  594         msg_len = len(messages[0])
  595         if msg_len < min_field_size:
! 596             return []
  597 
  598         # Compute positional entropy and variance
  599         position_entropy = np.zeros(msg_len)
  600         position_variance = np.zeros(msg_len)
```


---


Lines 623-631

```python
  623                         # All messages have same value
  624                         constant_val = messages[0][field_start:pos]
  625                         field_type = "constant"
  626                     elif avg_entropy > 6.0:
! 627                         field_type = "high_entropy"
  628                     else:
  629                         field_type = "variable"
  630 
  631                     fields.append(
```


---


Lines 661-669

```python
  661 
  662     def _detect_length_prefix(self, messages: list[bytes]) -> int | None:
  663         """Detect length prefix at start of messages."""
  664         if len(messages) < 3:
! 665             return None
  666 
  667         # Try 1-byte length at offset 0
  668         if all(len(msg) > 1 for msg in messages):
  669             if all(msg[0] == len(msg) for msg in messages):
```


---


Lines 674-684

```python
  674             matches = 0
  675             for msg in messages:
  676                 length_field = int.from_bytes(msg[0:2], byteorder="little")
  677                 if length_field == len(msg):
! 678                     matches += 1
  679             if matches / len(messages) > 0.8:
! 680                 return 0
  681 
  682         return None
  683 
  684     def _detect_checksum_field(self, messages: list[bytes]) -> int | None:
```


---


Lines 685-693

```python
  685         """Detect checksum field (simplified heuristic)."""
  686         # This is a simplified version - real implementation would test CRC8/16/32
  687         # For now, just detect if last 1-4 bytes have high variance (likely checksum)
  688         if len(messages) < 3:
! 689             return None
  690 
  691         msg_len = len(messages[0])
  692         if not all(len(msg) == msg_len for msg in messages):
  693             return None
```


---


Lines 701-714

```python
  701                 # If almost all unique, likely a checksum
  702                 if unique_values / len(messages) > 0.9:
  703                     return offset
  704 
! 705         return None
  706 
  707     def _shannon_entropy_bytes(self, data: bytes) -> float:
  708         """Calculate Shannon entropy for byte sequence."""
  709         if not data:
! 710             return 0.0
  711 
  712         byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
  713         probabilities = byte_counts[byte_counts > 0] / len(data)
  714         return float(-np.sum(probabilities * np.log2(probabilities)))
```


---


Lines 752-760

```python
  752             confidence += 0.2
  753 
  754         # More confidence if delimiter found
  755         if has_delimiter:
! 756             confidence += 0.1
  757 
  758         return min(1.0, float(confidence))
  759 
```


---


Lines 917-935

```python
  917     region_start = 0
  918 
  919     for offset, entropy in windows:
  920         if entropy > threshold:
! 921             if not in_region:
! 922                 region_start = offset
! 923                 in_region = True
  924         else:
  925             if in_region:
! 926                 regions.append((region_start, offset))
! 927                 in_region = False
  928 
  929     # Close final region
  930     if in_region:
! 931         regions.append((region_start, len(data)))
  932 
  933     return regions
  934 
```


---


Lines 960-977

```python
  960 
  961     for offset, entropy in windows:
  962         # Compressed: 6.5-7.5 bits/byte
  963         if 6.5 < entropy < 7.5:
! 964             if not in_region:
! 965                 region_start = offset
! 966                 in_region = True
  967         else:
  968             if in_region:
! 969                 regions.append((region_start, offset))
! 970                 in_region = False
  971 
  972     if in_region:
! 973         regions.append((region_start, len(data)))
  974 
  975     return regions
  976 
```


---


## src/oscura/analyzers/statistics/basic&#46;py

Lines 354-366

```python
  354         >>> # Get flat values without units
  355         >>> results = measure(trace, include_units=False)
  356         >>> mean_value = results["mean"]  # Just the float
  357     """
! 358     data = trace.data if isinstance(trace, WaveformTrace) else trace
  359 
  360     # Define unit mappings for statistical measurements
  361     # For generic signals we use voltage units, but this could be parameterized
! 362     unit_map = {
  363         "mean": "V",
  364         "variance": "V²",
  365         "std": "V",
  366         "min": "V",
```


---


Lines 376-404

```python
  376         "p99": "dimensionless",
  377     }
  378 
  379     # Get basic stats
! 380     basic = basic_stats(trace)
  381 
  382     # Get percentiles
! 383     percentile_values = percentiles(data, [1, 5, 25, 50, 75, 95, 99])
  384 
  385     # Combine into single dict
! 386     all_measurements = {**basic, **percentile_values}
  387 
  388     # Select requested measurements or all
! 389     if parameters is not None:
! 390         all_measurements = {k: v for k, v in all_measurements.items() if k in parameters}
  391 
  392     # Format results
! 393     if include_units:
! 394         results = {}
! 395         for name, value in all_measurements.items():
! 396             unit = unit_map.get(name, "")
! 397             results[name] = {"value": value, "unit": unit}
! 398         return results
  399     else:
! 400         return all_measurements
  401 
  402 
  403 __all__ = [
  404     "basic_stats",
```


---


## src/oscura/analyzers/waveform/measurements&#46;py

Lines 368-376

```python
  368 
  369     # Verify peak is significant (SNR check)
  370     # If the peak is not at least 3x the mean, it's likely noise
  371     if fft_mag[peak_idx] < 3.0 * np.mean(fft_mag[1:]):
! 372         return np.nan
  373 
  374     # Calculate frequency from peak index
  375     freq_resolution = trace.metadata.sample_rate / n
  376     return float(peak_idx * freq_resolution)
```


---


Lines 424-432

```python
  424         return np.nan
  425 
  426     # Convert boolean data to float if needed
  427     if data.dtype == bool:
! 428         data = data.astype(np.float64)
  429 
  430     low, high = _find_levels(data)
  431     amplitude = high - low
```


---


Lines 442-450

```python
  442     samples_high = np.sum(above_threshold)
  443     total_samples = len(data)
  444 
  445     if total_samples == 0:
! 446         return np.nan
  447 
  448     dc = float(samples_high) / total_samples
  449 
  450     if percentage:
```


---


Lines 936-944

```python
  936     # Sanity check - if histogram method failed, use adaptive percentiles
  937     if high <= low:
  938         # For extreme duty cycles, use min/max with small outlier rejection
  939         # p01 and p99 remove top/bottom 1% outliers (noise, ringing)
! 940         return float(p01), float(p99)
  941 
  942     return float(low), float(high)
  943 
```


---


## src/oscura/analyzers/waveform/spectral&#46;py

Lines 704-712

```python
  704     thd_ratio = np.sqrt(harmonic_power) / fund_mag
  705 
  706     # Validate: THD must always be non-negative
  707     if thd_ratio < 0:
! 708         raise ValueError(
  709             f"THD ratio is negative ({thd_ratio:.6f}), indicating a calculation error. "
  710             f"Fundamental: {fund_mag:.6f}, Harmonic power: {harmonic_power:.6f}"
  711         )
```


---


Lines 2065-2073

```python
  2065         fundamental_freq = fund_freq
  2066 
  2067     if fundamental_freq == 0:
  2068         # Return empty result
! 2069         return {
  2070             "frequencies": np.array([]),
  2071             "amplitudes": np.array([]),
  2072             "amplitudes_db": np.array([]),
  2073             "fundamental_freq": np.array([0.0]),
```


---


Lines 2079-2092

```python
  2079 
  2080     for h in range(1, n_harmonics + 2):  # Include fundamental (h=1)
  2081         target_freq = h * fundamental_freq
  2082         if target_freq > freq[-1]:
! 2083             break
  2084 
  2085         # Search around expected frequency
  2086         search_mask = np.abs(freq - target_freq) <= search_width_hz
  2087         if not np.any(search_mask):
! 2088             continue
  2089 
  2090         # Find peak in search region
  2091         search_region_mag = magnitude.copy()
  2092         search_region_mag[~search_mask] = 0
```


---


Lines 2235-2243

```python
  2235     References:
  2236         IEEE 1241-2010: ADC Terminology and Test Methods
  2237     """
  2238     # Define all available spectral measurements with units
! 2239     all_measurements = {
  2240         "thd": (thd, "%"),
  2241         "snr": (snr, "dB"),
  2242         "sinad": (sinad, "dB"),
  2243         "enob": (enob, "bits"),
```


---


Lines 2244-2288

```python
  2244         "sfdr": (sfdr, "dB"),
  2245     }
  2246 
  2247     # Select requested measurements or all
! 2248     if parameters is None:
! 2249         selected = all_measurements
  2250     else:
! 2251         selected = {k: v for k, v in all_measurements.items() if k in parameters}
  2252 
! 2253     results: dict[str, Any] = {}
  2254 
! 2255     for name, (func, unit) in selected.items():
! 2256         try:
! 2257             value = func(trace)  # type: ignore[operator]
! 2258         except Exception:
! 2259             value = np.nan
  2260 
! 2261         if include_units:
! 2262             results[name] = {"value": value, "unit": unit}
  2263         else:
! 2264             results[name] = value
  2265 
  2266     # Add dominant frequency if requested or if computing all
! 2267     if parameters is None or "dominant_freq" in parameters:
! 2268         try:
! 2269             fft_result = fft(trace, return_phase=False)
! 2270             freq, magnitude = fft_result[0], fft_result[1]
! 2271             dominant_idx = int(np.argmax(np.abs(magnitude[1:]))) + 1  # Skip DC
! 2272             dominant_freq_value = float(freq[dominant_idx])
  2273 
! 2274             if include_units:
! 2275                 results["dominant_freq"] = {"value": dominant_freq_value, "unit": "Hz"}
  2276             else:
! 2277                 results["dominant_freq"] = dominant_freq_value
! 2278         except Exception:
! 2279             if include_units:
! 2280                 results["dominant_freq"] = {"value": np.nan, "unit": "Hz"}
  2281             else:
! 2282                 results["dominant_freq"] = np.nan
  2283 
! 2284     return results
  2285 
  2286 
  2287 __all__ = [
  2288     "bartlett_psd",
```


---


## src/oscura/automotive/**init**&#46;py

Lines 48-56

```python
  48 
  49     __version__ = version("oscura")
  50 except Exception:
  51     # Fallback for development/testing when package not installed
! 52     __version__ = "0.8.0"
  53 
  54 __all__ = [
  55     "CANMessage",
  56     "CANSession",
```


---


## src/oscura/reporting/automation&#46;py

Lines 141-149

```python
  141     mean = np.mean(values)
  142     std = np.std(values)
  143 
  144     if std == 0:
! 145         return anomalies
  146 
  147     # Check each measurement
  148     for name, value in numeric_measurements.items():
  149         z_score = abs((value - mean) / std)
```


---


Lines 161-169

```python
  161 
  162     # Domain-specific anomaly checks
  163     for name, value in measurements.items():
  164         if not isinstance(value, int | float):
! 165             continue
  166 
  167         # Negative SNR
  168         if "snr" in name.lower() and value < 0:
  169             anomalies.append(
```


---


Lines 256-268

```python
  256         suggestions.append("Perform power quality analysis (harmonics, flicker, sags/swells)")
  257 
  258     # Eye diagram quality issues
  259     if interpretations:
! 260         poor_quality = [
  261             k for k, v in interpretations.items() if v.quality.value in ("marginal", "poor")
  262         ]
! 263         if poor_quality:
! 264             suggestions.append(
  265                 f"Investigate {len(poor_quality)} marginal/poor measurements: "
  266                 f"{', '.join(poor_quality[:3])}"
  267             )
```


---


Lines 336-348

```python
  336     flat = {}
  337 
  338     for key, value in results.items():
  339         if isinstance(value, dict):
! 340             if "value" in value:
! 341                 flat[key] = value["value"]
  342             else:
! 343                 nested = _flatten_results(value)
! 344                 flat.update(nested)
  345         elif isinstance(value, int | float | str):
  346             flat[key] = value
  347 
  348     return flat
```


---


## src/oscura/reporting/citations&#46;py

Lines 250-258

```python
  250 
  251             if url:
  252                 lines.append(f'<li><a href="{url}" target="_blank">{bib}</a></li>')
  253             else:
! 254                 lines.append(f"<li>{bib}</li>")
  255 
  256         lines.extend(["</ol>", "</div>"])
  257         return "\n".join(lines)
```


---


Lines 368-375

```python
  368     # Oscilloscope/digitizer -> IEEE 1057
  369     if any(
  370         term in measurement_lower for term in ["bandwidth", "sample_rate", "resolution", "accuracy"]
  371     ):
! 372         return "1057"
  373 
  374     return None
```


---


## src/oscura/reporting/formatting/measurements&#46;py

Lines 40-48

```python
  40 
  41     def __post_init__(self) -> None:
  42         """Initialize number formatter if not provided."""
  43         if self.number_formatter is None:
! 44             self.number_formatter = NumberFormatter(sig_figs=self.default_sig_figs)
  45 
  46     def format_single(self, value: float, unit: str = "") -> str:
  47         """Format single measurement value with SI prefix and unit.
```


---


Lines 72-86

```python
  72             return f"{formatted} %" if self.show_units else formatted
  73 
  74         # Handle percentages (THD, etc.) that are already in percentage units
  75         if unit == "%":
! 76             formatted = self.number_formatter.format(value, "")
! 77             return f"{formatted} %" if self.show_units else formatted
  78 
  79         # Handle dimensionless values
  80         if unit == "":
! 81             formatted = self.number_formatter.format(value, "")
! 82             return formatted
  83 
  84         # Use NumberFormatter's built-in SI prefix support for other units
  85         formatted = self.number_formatter.format(value, unit)
  86         return formatted if self.show_units else formatted.replace(unit, "").strip()
```


---


Lines 114-125

```python
  114         value = measurement.get("value")
  115         unit = measurement.get("unit", "")
  116 
  117         if value is None:
! 118             return "N/A"
  119 
  120         if not isinstance(value, (int, float)):
! 121             return str(value)
  122 
  123         # Format the value
  124         formatted = self.format_single(value, unit)
```


---


Lines 124-145

```python
  124         formatted = self.format_single(value, unit)
  125 
  126         # Add spec comparison if requested
  127         if self.show_specs and "spec" in measurement:
! 128             spec = measurement["spec"]
! 129             spec_type = measurement.get("spec_type", "exact")
! 130             spec_formatted = self.format_single(spec, unit)
  131 
! 132             if spec_type == "max":
! 133                 formatted += f" (spec: < {spec_formatted})"
! 134             elif spec_type == "min":
! 135                 formatted += f" (spec: > {spec_formatted})"
  136             else:  # exact
! 137                 formatted += f" (spec: {spec_formatted})"
  138 
  139             # Add pass/fail indicator
! 140             if "passed" in measurement:
! 141                 formatted += " ✓" if measurement["passed"] else " ✗"
  142 
  143         return formatted
  144 
  145     def format_measurement_dict(
```


---


Lines 172-185

```python
  172 
  173             if html:
  174                 lines.append(f"<li><strong>{display_name}:</strong> {formatted_value}</li>")
  175             else:
! 176                 lines.append(f"{display_name}: {formatted_value}")
  177 
  178         if html:
  179             return f"<ul>\n{''.join(lines)}\n</ul>"
  180         else:
! 181             return "\n".join(lines)
  182 
  183     def to_display_dict(self, measurements: dict[str, dict[str, Any]]) -> dict[str, str]:
  184         """Convert measurements to display-ready string dictionary.
```


---


Lines 194-202

```python
  194             >>> measurements = {"rise_time": {"value": 2.3e-9, "unit": "s"}}
  195             >>> formatter.to_display_dict(measurements)
  196             {'rise_time': '2.30 ns'}
  197         """
! 198         return {key: self.format_measurement(meas) for key, meas in measurements.items()}
  199 
  200 
  201 def format_measurement(measurement: dict[str, Any], sig_figs: int = 4) -> str:
  202     """Quick format single measurement dict.
```


---


Lines 214-223

```python
  214     Example:
  215         >>> format_measurement({"value": 2.3e-9, "unit": "s"})
  216         '2.300 ns'
  217     """
! 218     formatter = MeasurementFormatter(number_formatter=NumberFormatter(sig_figs=sig_figs))
! 219     return formatter.format_measurement(measurement)
  220 
  221 
  222 def format_measurement_dict(
  223     measurements: dict[str, dict[str, Any]], sig_figs: int = 4, html: bool = True
```


---


## src/oscura/reporting/html&#46;py

Lines 549-557

```python
  549     """Render section content (text, tables, figures)."""
  550     if isinstance(section.content, str):
  551         # Check if content is already HTML (starts with HTML tag)
  552         if section.content.strip().startswith("<"):
! 553             return section.content  # Return HTML as-is
  554         return f"<p>{section.content}</p>"
  555 
  556     if isinstance(section.content, list):
  557         return _render_content_list(section.content)
```


---


Lines 676-684

```python
  676 
  677     for plot_name, plot_data in plots.items():
  678         # Ensure data URI format
  679         if not plot_data.startswith("data:"):
! 680             plot_data = f"data:image/png;base64,{plot_data}"
  681 
  682         plot_html += f"\n<h3>{plot_name.replace('_', ' ').title()}</h3>\n"
  683         plot_html += (
  684             f'<img src="{plot_data}" alt="{plot_name}" '
```


---


Lines 691-699

```python
  691     # Insert at specified location
  692     if insert_location == "before_closing_div" and "</div>" in html_content:
  693         html_content = html_content.replace("</div>", f"{plot_html}</div>", 1)
  694     elif insert_location == "before_closing_body" and "</body>" in html_content:
! 695         html_content = html_content.replace("</body>", f"{plot_html}</body>")
  696     else:
  697         # Append to end
  698         html_content += plot_html
```


---


## src/oscura/reporting/interpretation&#46;py

Lines 117-132

```python
  117 
  118     # Rise time interpretation
  119     if "rise" in name.lower() or "fall" in name.lower():
  120         if value < 1e-9:
! 121             interpretation = "Very fast transition time, indicating high bandwidth signal path"
! 122             quality = QualityLevel.EXCELLENT
  123         elif value < 10e-9:
  124             interpretation = "Fast transition time, suitable for high-speed digital signals"
  125             quality = QualityLevel.GOOD
  126         elif value < 100e-9:
! 127             interpretation = "Moderate transition time, adequate for standard digital signals"
! 128             quality = QualityLevel.ACCEPTABLE
  129         else:
  130             interpretation = "Slow transition time, may limit signal bandwidth"
  131             quality = QualityLevel.MARGINAL
  132             recommendations.append("Consider improving signal path bandwidth")
```


---


Lines 139-149

```python
  139         elif value > 40:
  140             interpretation = "Good signal-to-noise ratio, acceptable for most applications"
  141             quality = QualityLevel.GOOD
  142         elif value > 20:
! 143             interpretation = "Moderate SNR, noise may affect precision measurements"
! 144             quality = QualityLevel.ACCEPTABLE
! 145             recommendations.append("Consider noise reduction techniques")
  146         else:
  147             interpretation = "Poor SNR, signal quality compromised by noise"
  148             quality = QualityLevel.POOR
  149             recommendations.extend(
```


---


Lines 157-166

```python
  157         if jitter_ps < 10:
  158             interpretation = "Very low jitter, excellent timing stability"
  159             quality = QualityLevel.EXCELLENT
  160         elif jitter_ps < 50:
! 161             interpretation = "Low jitter, good timing performance"
! 162             quality = QualityLevel.GOOD
  163         elif jitter_ps < 200:
  164             interpretation = "Moderate jitter, acceptable for most applications"
  165             quality = QualityLevel.ACCEPTABLE
  166         else:
```


---


Lines 180-189

```python
  180                 f"Good bandwidth ({value / 1e6:.0f} MHz), adequate for most applications"
  181             )
  182             quality = QualityLevel.GOOD
  183         elif value > 10e6:
! 184             interpretation = f"Moderate bandwidth ({value / 1e6:.1f} MHz)"
! 185             quality = QualityLevel.ACCEPTABLE
  186         else:
  187             interpretation = f"Limited bandwidth ({value / 1e6:.2f} MHz)"
  188             quality = QualityLevel.MARGINAL
```


---


Lines 207-224

```python
  207 
  208                 if min_margin > 0.3:
  209                     interpretation = "Well within specification with good margin"
  210                     quality = QualityLevel.GOOD
! 211                 elif min_margin > 0.1:
! 212                     interpretation = "Within specification with adequate margin"
! 213                     quality = QualityLevel.ACCEPTABLE
  214                 else:
! 215                     interpretation = "Within specification but marginal"
! 216                     quality = QualityLevel.MARGINAL
! 217                     recommendations.append("Low margin to specification limits")
  218             else:
! 219                 interpretation = "Within specification"
! 220                 quality = QualityLevel.ACCEPTABLE
  221 
  222     else:
  223         interpretation = f"Measured value: {value} {units}"
  224         quality = QualityLevel.ACCEPTABLE
```


---


Lines 261-269

```python
  261     for name, value in measurements.items():
  262         if isinstance(value, float):
  263             description_parts.append(f"{name}: {value:.3f}")
  264         else:
! 265             description_parts.append(f"{name}: {value}")
  266 
  267     description = "\n".join(description_parts)
  268 
  269     recommendation = ""
```


---


## src/oscura/reporting/summary&#46;py

Lines 49-89

```python
  49         >>> summary = generate_executive_summary(measurements)
  50         >>> "Executive Summary" in summary or len(summary) > 0
  51         True
  52     """
! 53     sections = []
  54 
  55     # Overall status
! 56     status_section = _generate_status_section(measurements, interpretations)
! 57     sections.append(status_section)
  58 
  59     # Key findings
! 60     findings_section = _generate_findings_section(measurements, interpretations, max_findings)
! 61     sections.append(findings_section)
  62 
  63     # Recommendations
! 64     if interpretations:
! 65         rec_section = _generate_recommendations_section(interpretations)
! 66         if rec_section.bullet_points:
! 67             sections.append(rec_section)
  68 
  69     # Format as text
! 70     lines = ["# Executive Summary", ""]
  71 
! 72     for section in sections:
! 73         lines.append(f"## {section.title}")
! 74         lines.append("")
  75 
! 76         if section.content:
! 77             lines.append(section.content)
! 78             lines.append("")
  79 
! 80         if section.bullet_points:
! 81             for point in section.bullet_points:
! 82                 lines.append(f"- {point}")
! 83             lines.append("")
  84 
! 85     return "\n".join(lines)
  86 
  87 
  88 def _generate_status_section(
  89     measurements: dict[str, Any],
```


---


Lines 89-122

```python
   89     measurements: dict[str, Any],
   90     interpretations: dict[str, MeasurementInterpretation] | None,
   91 ) -> ExecutiveSummarySection:
   92     """Generate overall status section."""
!  93     total = len(measurements)
   94 
!  95     if interpretations:
!  96         excellent = sum(1 for i in interpretations.values() if i.quality == QualityLevel.EXCELLENT)
!  97         good = sum(1 for i in interpretations.values() if i.quality == QualityLevel.GOOD)
!  98         acceptable = sum(
   99             1 for i in interpretations.values() if i.quality == QualityLevel.ACCEPTABLE
  100         )
! 101         marginal = sum(1 for i in interpretations.values() if i.quality == QualityLevel.MARGINAL)
! 102         poor = sum(1 for i in interpretations.values() if i.quality == QualityLevel.POOR)
! 103         failed = sum(1 for i in interpretations.values() if i.quality == QualityLevel.FAILED)
  104 
! 105         if failed > 0:
! 106             overall = "CRITICAL"
! 107             content = f"{failed} of {total} measurements failed critical requirements."
! 108         elif marginal > total / 2:
! 109             overall = "MARGINAL"
! 110             content = f"{marginal} of {total} measurements are marginal."
! 111         elif excellent + good > total * 0.7:
! 112             overall = "GOOD"
! 113             content = f"Signal quality is good: {excellent + good} of {total} measurements are excellent or good."
  114         else:
! 115             overall = "ACCEPTABLE"
! 116             content = f"{acceptable} of {total} measurements are acceptable."
  117 
! 118         bullet_points = [
  119             f"Excellent: {excellent}",
  120             f"Good: {good}",
  121             f"Acceptable: {acceptable}",
  122             f"Marginal: {marginal}",
```


---


Lines 123-135

```python
  123             f"Poor: {poor}",
  124             f"Failed: {failed}",
  125         ]
  126     else:
! 127         overall = "COMPLETE"
! 128         content = f"Analysis complete with {total} measurements."
! 129         bullet_points = []
  130 
! 131     return ExecutiveSummarySection(
  132         title=f"Overall Status: {overall}",
  133         content=content,
  134         bullet_points=bullet_points,
  135         priority=1,
```


---


Lines 141-151

```python
  141     interpretations: dict[str, MeasurementInterpretation] | None,
  142     max_findings: int,
  143 ) -> ExecutiveSummarySection:
  144     """Generate key findings section."""
! 145     findings = identify_key_findings(measurements, interpretations, max_findings)
  146 
! 147     return ExecutiveSummarySection(
  148         title="Key Findings",
  149         content="",
  150         bullet_points=findings,
  151         priority=2,
```


---


Lines 155-171

```python
  155 def _generate_recommendations_section(
  156     interpretations: dict[str, MeasurementInterpretation],
  157 ) -> ExecutiveSummarySection:
  158     """Generate recommendations section."""
! 159     all_recommendations = []
  160 
! 161     for interp in interpretations.values():
! 162         all_recommendations.extend(interp.recommendations)
  163 
  164     # Deduplicate
! 165     unique_recs = list(dict.fromkeys(all_recommendations))
  166 
! 167     return ExecutiveSummarySection(
  168         title="Recommendations",
  169         content="",
  170         bullet_points=unique_recs[:5],  # Top 5
  171         priority=3,
```


---


Lines 245-253

```python
  245             for name, interp in interpretations.items()
  246             if interp.quality in (QualityLevel.FAILED, QualityLevel.POOR)
  247         ]
  248         if failed:
! 249             findings.append(f"Critical: {len(failed)} measurements failed or poor quality")
  250 
  251         # Excellent quality
  252         excellent = [
  253             name
```


---


Lines 254-262

```python
  254             for name, interp in interpretations.items()
  255             if interp.quality == QualityLevel.EXCELLENT
  256         ]
  257         if excellent:
! 258             findings.append(f"{len(excellent)} measurements show excellent performance")
  259 
  260     # Domain-specific findings
  261     if "snr" in measurements:
  262         snr = measurements["snr"]
```


---


Lines 261-271

```python
  261     if "snr" in measurements:
  262         snr = measurements["snr"]
  263         if isinstance(snr, int | float):
  264             if snr > 60:
! 265                 findings.append(f"Excellent SNR: {snr:.1f} dB")
  266             elif snr < 20:
! 267                 findings.append(f"Low SNR: {snr:.1f} dB - noise mitigation recommended")
  268 
  269     if "bandwidth" in measurements:
  270         bw = measurements["bandwidth"]
  271         if isinstance(bw, int | float):
```


---


Lines 269-287

```python
  269     if "bandwidth" in measurements:
  270         bw = measurements["bandwidth"]
  271         if isinstance(bw, int | float):
  272             if bw > 1e9:
! 273                 findings.append(f"Wide bandwidth: {bw / 1e9:.2f} GHz")
  274 
  275     if "jitter" in measurements or "rms_jitter" in measurements:
! 276         jitter_key = "rms_jitter" if "rms_jitter" in measurements else "jitter"
! 277         jitter = measurements[jitter_key]
! 278         if isinstance(jitter, int | float):
! 279             jitter_ps = jitter * 1e12
! 280             if jitter_ps < 10:
! 281                 findings.append("Excellent timing: RMS jitter < 10 ps")
! 282             elif jitter_ps > 200:
! 283                 findings.append(f"High jitter: {jitter_ps:.1f} ps - investigate timing issues")
  284 
  285     # Limit to max_findings
  286     return findings[:max_findings]
```


---


Lines 317-330

```python
  317     # Add domain-specific recommendations
  318     if "snr" in measurements:
  319         snr = measurements["snr"]
  320         if isinstance(snr, int | float) and snr < 30:
! 321             recommendations.append("Investigate noise sources and consider filtering")
  322 
  323     if "bandwidth" in measurements:
  324         bw = measurements["bandwidth"]
  325         if isinstance(bw, int | float) and bw < 100e6:
! 326             recommendations.append("Verify signal path bandwidth requirements")
  327 
  328     # Deduplicate and return
  329     return list(dict.fromkeys(recommendations))
```


---


## src/oscura/reporting/visualization&#46;py

Lines 69-101

```python
   69         Example:
   70             >>> fig, ax = plt.subplots()
   71             >>> PlotStyler.apply_ieee_style(ax, "Time", "Voltage", "Signal")
   72         """
!  73         if not HAS_MATPLOTLIB:
!  74             return
   75 
   76         # Labels and title
!  77         if xlabel:
!  78             ax.set_xlabel(xlabel, fontsize=10, fontweight="normal")
!  79         if ylabel:
!  80             ax.set_ylabel(ylabel, fontsize=10, fontweight="normal")
!  81         if title:
!  82             ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
   83 
   84         # Grid
!  85         if grid:
!  86             ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color=IEEE_COLORS["grid"])
   87 
   88         # Spines
!  89         for spine in ax.spines.values():
!  90             spine.set_color(IEEE_COLORS["text"])
!  91             spine.set_linewidth(0.8)
   92 
   93         # Ticks
!  94         ax.tick_params(labelsize=9, colors=IEEE_COLORS["text"])
   95 
   96         # Tight layout
!  97         ax.figure.tight_layout()
   98 
   99 
  100 class IEEEPlotGenerator:
  101     """Generate IEEE-compliant plots for signal analysis reports.
```


---


Lines 118-131

```python
  118 
  119         Raises:
  120             ImportError: If matplotlib is not installed.
  121         """
! 122         if not HAS_MATPLOTLIB:
! 123             raise ImportError("matplotlib is required for plot generation")
  124 
! 125         self.dpi = dpi
! 126         self.figsize = figsize
! 127         self.styler = PlotStyler()
  128 
  129     def plot_waveform(
  130         self,
  131         time: NDArray[np.floating[Any]],
```


---


Lines 152-169

```python
  152             >>> t = np.linspace(0, 1, 1000)
  153             >>> s = np.sin(2 * np.pi * 10 * t)
  154             >>> fig = generator.plot_waveform(t, s, "Sine Wave")
  155         """
! 156         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  157 
  158         # Plot signal
! 159         ax.plot(time, signal, color=IEEE_COLORS["primary"], linewidth=1.5, label="Signal")
  160 
  161         # Add markers if provided
! 162         if markers:
! 163             for label, pos in markers.items():
! 164                 ax.axvline(pos, color=IEEE_COLORS["accent"], linestyle="--", alpha=0.7)
! 165                 ax.text(
  166                     pos,
  167                     ax.get_ylim()[1] * 0.9,
  168                     label,
  169                     rotation=90,
```


---


Lines 170-179

```python
  170                     verticalalignment="top",
  171                     fontsize=8,
  172                 )
  173 
! 174         self.styler.apply_ieee_style(ax, xlabel, ylabel, title)
! 175         return fig
  176 
  177     def plot_fft(
  178         self,
  179         frequencies: NDArray[np.floating[Any]],
```


---


Lines 196-222

```python
  196             >>> freq = np.fft.rfftfreq(1000, 1/1000)
  197             >>> mag_db = 20 * np.log10(np.abs(np.fft.rfft(signal)))
  198             >>> fig = generator.plot_fft(freq, mag_db)
  199         """
! 200         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  201 
  202         # Plot spectrum
! 203         ax.plot(frequencies, magnitude_db, color=IEEE_COLORS["primary"], linewidth=1.5)
  204 
  205         # Mark peaks
! 206         if peak_markers > 0:
  207             # Find peaks (ignore DC)
! 208             valid_idx = frequencies > 0
! 209             valid_freq = frequencies[valid_idx]
! 210             valid_mag = magnitude_db[valid_idx]
  211 
! 212             if len(valid_mag) > 0:
! 213                 peak_indices = np.argsort(valid_mag)[-peak_markers:]
! 214                 for idx in peak_indices:
! 215                     freq_val = valid_freq[idx]
! 216                     mag_val = valid_mag[idx]
! 217                     ax.plot(freq_val, mag_val, "ro", markersize=6, alpha=0.7)
! 218                     ax.annotate(
  219                         f"{freq_val:.1f} Hz",
  220                         (freq_val, mag_val),
  221                         xytext=(5, 5),
  222                         textcoords="offset points",
```


---


Lines 223-237

```python
  223                         fontsize=8,
  224                         color=IEEE_COLORS["danger"],
  225                     )
  226 
! 227         self.styler.apply_ieee_style(ax, "Frequency (Hz)", "Magnitude (dB)", title)
  228 
  229         # Log scale for frequency if range > 2 decades
! 230         if len(frequencies) > 1 and frequencies[-1] / frequencies[1] > 100:
! 231             ax.set_xscale("log")
  232 
! 233         return fig
  234 
  235     def plot_psd(
  236         self,
  237         frequencies: NDArray[np.floating[Any]],
```


---


Lines 254-274

```python
  254             >>> from scipy import signal as sp_signal
  255             >>> freq, psd = sp_signal.welch(data, fs=sample_rate)
  256             >>> fig = generator.plot_psd(freq, psd)
  257         """
! 258         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  259 
  260         # Convert to dB scale
! 261         psd_db = 10 * np.log10(psd + 1e-12)  # Add epsilon to avoid log(0)
  262 
! 263         ax.plot(frequencies, psd_db, color=IEEE_COLORS["primary"], linewidth=1.5)
! 264         self.styler.apply_ieee_style(ax, "Frequency (Hz)", f"PSD (dB {units})", title)
  265 
  266         # Log scale for frequency
! 267         if len(frequencies) > 1 and frequencies[-1] / frequencies[1] > 100:
! 268             ax.set_xscale("log")
  269 
! 270         return fig
  271 
  272     def plot_spectrogram(
  273         self,
  274         time: NDArray[np.floating[Any]],
```


---


Lines 291-315

```python
  291             >>> from scipy import signal as sp_signal
  292             >>> f, t, Sxx = sp_signal.spectrogram(data, fs=sample_rate)
  293             >>> fig = generator.plot_spectrogram(t, f, Sxx)
  294         """
! 295         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  296 
  297         # Convert to dB scale
! 298         spec_db = 10 * np.log10(spectrogram + 1e-12)
  299 
  300         # Plot spectrogram
! 301         im = ax.pcolormesh(
  302             time, frequencies, spec_db, shading="auto", cmap="viridis", rasterized=True
  303         )
  304 
  305         # Colorbar
! 306         cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
! 307         cbar.ax.tick_params(labelsize=9)
  308 
! 309         self.styler.apply_ieee_style(ax, "Time (s)", "Frequency (Hz)", title, grid=False)
  310 
! 311         return fig
  312 
  313     def plot_eye_diagram(
  314         self,
  315         signal: NDArray[np.floating[Any]],
```


---


Lines 331-350

```python
  331         Example:
  332             >>> # For 1000 samples at 10 samples/symbol
  333             >>> fig = generator.plot_eye_diagram(signal, 10, num_traces=50)
  334         """
! 335         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  336 
  337         # Extract symbol windows
! 338         num_symbols = len(signal) // samples_per_symbol
! 339         num_traces = min(num_traces, num_symbols - 1)
  340 
! 341         for i in range(num_traces):
! 342             start = i * samples_per_symbol
! 343             end = start + 2 * samples_per_symbol  # Two symbol periods
! 344             if end <= len(signal):
! 345                 trace = signal[start:end]
! 346                 ax.plot(
  347                     np.arange(len(trace)),
  348                     trace,
  349                     color=IEEE_COLORS["primary"],
  350                     alpha=0.3,
```


---


Lines 350-359

```python
  350                     alpha=0.3,
  351                     linewidth=0.5,
  352                 )
  353 
! 354         self.styler.apply_ieee_style(ax, "Sample", "Amplitude", title)
! 355         return fig
  356 
  357     def plot_histogram(
  358         self,
  359         data: NDArray[np.floating[Any]],
```


---


Lines 374-393

```python
  374 
  375         Example:
  376             >>> fig = generator.plot_histogram(signal, bins=100, title="Voltage Distribution")
  377         """
! 378         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
  379 
  380         # Plot histogram
! 381         n, bins_edges, _ = ax.hist(
  382             data, bins=bins, color=IEEE_COLORS["primary"], alpha=0.7, edgecolor="black"
  383         )
  384 
  385         # Fit Gaussian
! 386         mu = np.mean(data)
! 387         sigma = np.std(data)
! 388         x = np.linspace(bins_edges[0], bins_edges[-1], 200)
! 389         gaussian = (
  390             len(data)
  391             * (bins_edges[1] - bins_edges[0])
  392             / (sigma * np.sqrt(2 * np.pi))
  393             * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
```


---


Lines 391-403

```python
  391             * (bins_edges[1] - bins_edges[0])
  392             / (sigma * np.sqrt(2 * np.pi))
  393             * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
  394         )
! 395         ax.plot(x, gaussian, color=IEEE_COLORS["danger"], linewidth=2, label="Gaussian fit")
  396 
  397         # Add statistics text
! 398         stats_text = f"mean = {mu:.4f}\nstd = {sigma:.4f}"
! 399         ax.text(
  400             0.98,
  401             0.98,
  402             stats_text,
  403             transform=ax.transAxes,
```


---


Lines 406-416

```python
  406             horizontalalignment="right",
  407             bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
  408         )
  409 
! 410         ax.legend(fontsize=9)
! 411         self.styler.apply_ieee_style(ax, xlabel, "Count", title)
! 412         return fig
  413 
  414     def plot_jitter(
  415         self,
  416         time_intervals: NDArray[np.floating[Any]],
```


---


Lines 428-441

```python
  428         Example:
  429             >>> # time_intervals in seconds
  430             >>> fig = generator.plot_jitter(time_intervals)
  431         """
! 432         fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
! 433         gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
  434 
  435         # Time series plot
! 436         ax1 = fig.add_subplot(gs[0])
! 437         ax1.plot(
  438             np.arange(len(time_intervals)),
  439             time_intervals * 1e9,  # Convert to nanoseconds
  440             color=IEEE_COLORS["primary"],
  441             linewidth=1,
```


---


Lines 442-465

```python
  442             marker="o",
  443             markersize=2,
  444             alpha=0.6,
  445         )
! 446         self.styler.apply_ieee_style(ax1, "Interval #", "Jitter (ns)", f"{title} - Time Series")
  447 
  448         # Histogram
! 449         ax2 = fig.add_subplot(gs[1])
! 450         jitter_ns = time_intervals * 1e9
! 451         ax2.hist(jitter_ns, bins=50, color=IEEE_COLORS["secondary"], alpha=0.7, edgecolor="black")
  452 
  453         # Statistics
! 454         mean_jitter = np.mean(jitter_ns)
! 455         std_jitter = np.std(jitter_ns)
! 456         pk_pk_jitter = np.max(jitter_ns) - np.min(jitter_ns)
  457 
! 458         stats_text = (
  459             f"Mean: {mean_jitter:.3f} ns\nStd: {std_jitter:.3f} ns\nPk-Pk: {pk_pk_jitter:.3f} ns"
  460         )
! 461         ax2.text(
  462             0.98,
  463             0.98,
  464             stats_text,
  465             transform=ax2.transAxes,
```


---


Lines 468-478

```python
  468             horizontalalignment="right",
  469             bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
  470         )
  471 
! 472         self.styler.apply_ieee_style(ax2, "Jitter (ns)", "Count", "Distribution")
  473 
! 474         return fig
  475 
  476     def plot_power(
  477         self,
  478         time: NDArray[np.floating[Any]],
```


---


Lines 493-520

```python
  493 
  494         Example:
  495             >>> fig = generator.plot_power(time, voltage, current)
  496         """
! 497         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)
  498 
  499         # Voltage
! 500         ax1.plot(time, voltage, color=IEEE_COLORS["primary"], linewidth=1.5)
! 501         self.styler.apply_ieee_style(ax1, "", "Voltage (V)", "Voltage", grid=True)
  502 
  503         # Current
! 504         ax2.plot(time, current, color=IEEE_COLORS["secondary"], linewidth=1.5)
! 505         self.styler.apply_ieee_style(ax2, "", "Current (A)", "Current", grid=True)
  506 
  507         # Power
! 508         power = voltage * current
! 509         ax3.plot(time, power, color=IEEE_COLORS["accent"], linewidth=1.5)
! 510         ax3.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
! 511         self.styler.apply_ieee_style(ax3, "Time (s)", "Power (W)", "Instantaneous Power", grid=True)
  512 
! 513         fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
! 514         fig.tight_layout()
  515 
! 516         return fig
  517 
  518     @staticmethod
  519     def figure_to_base64(fig: Figure, format: str = "png") -> str:
  520         """Convert matplotlib figure to base64-encoded string for HTML embedding.
```


---


Lines 531-543

```python
  531             >>> img_str = IEEEPlotGenerator.figure_to_base64(fig)
  532             >>> "data:image/png;base64," in img_str
  533             True
  534         """
! 535         buffer = BytesIO()
! 536         fig.savefig(buffer, format=format, bbox_inches="tight", dpi=150)
! 537         buffer.seek(0)
! 538         img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
! 539         buffer.close()
! 540         plt.close(fig)
  541 
! 542         return f"data:image/{format};base64,{img_base64}"
```


---


## src/oscura/visualization/batch&#46;py

Lines 95-105

```python
   95 
   96     # Prepare data with downsampling if needed
   97     data = trace.data
   98     if len(data) > sample_limit:
!  99         step = len(data) // sample_limit
! 100         data = data[::step]
! 101         time = np.arange(len(data)) * step / trace.metadata.sample_rate
  102     else:
  103         time = np.arange(len(data)) / trace.metadata.sample_rate
  104 
  105     # Plot waveform
```


---


Lines 328-338

```python
  328 
  329     # Prepare data with downsampling if needed
  330     data = trace.data.astype(float)
  331     if len(data) > max_samples:
! 332         step = len(data) // max_samples
! 333         data = data[::step]
! 334         time = np.arange(len(data)) * step / trace.metadata.sample_rate
  335     else:
  336         time = np.arange(len(data)) / trace.metadata.sample_rate
  337 
  338     # Plot as step function (digital signal)
```


---


Lines 388-414

```python
  388         whiskerprops={"color": "black", "linewidth": 1.5},
  389         capprops={"color": "black", "linewidth": 1.5},
  390     )
  391 
! 392     ax1.set_ylabel("Amplitude (V)", fontsize=11, fontweight="bold")
! 393     ax1.set_title("Box Plot", fontsize=12, fontweight="bold")
! 394     ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
! 395     ax1.set_xticks([])
  396 
  397     # Violin plot
! 398     parts = ax2.violinplot([data], vert=True, widths=0.7, showmeans=True, showextrema=True)
! 399     for pc in parts["bodies"]:
! 400         pc.set_facecolor(COLORS["success"])
! 401         pc.set_alpha(0.7)
  402 
! 403     ax2.set_ylabel("Amplitude (V)", fontsize=11, fontweight="bold")
! 404     ax2.set_title("Violin Plot", fontsize=12, fontweight="bold")
! 405     ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
! 406     ax2.set_xticks([])
  407 
! 408     fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
! 409     plt.tight_layout()
! 410     return fig_to_base64(fig)
  411 
  412 
  413 def generate_all_plots(
  414     trace: WaveformTrace | DigitalTrace,
```


---


Lines 439-447

```python
  439         >>> len(plots)  # 5 plots for analog signal
  440         5
  441     """
  442     if output_format != "base64":
! 443         raise ValueError(f"Only 'base64' output format supported, got '{output_format}'")
  444 
  445     plots = {}
  446     is_digital = (
  447         trace.is_digital if hasattr(trace, "is_digital") else isinstance(trace, DigitalTrace)
```


---


Lines 450-461

```python
  450     # Always generate waveform plot
  451     try:
  452         plots["waveform"] = plot_waveform(trace)
  453         if verbose:
! 454             print("  ✓ Generated waveform plot")
! 455     except Exception as e:
! 456         if verbose:
! 457             print(f"  ⚠ Waveform plot failed: {e}")
  458 
  459     if not is_digital:
  460         # Analog signal plots
  461         if isinstance(trace, WaveformTrace):  # Type narrowing
```


---


Lines 461-496

```python
  461         if isinstance(trace, WaveformTrace):  # Type narrowing
  462             try:
  463                 plots["fft"] = plot_fft_spectrum(trace)
  464                 if verbose:
! 465                     print("  ✓ Generated FFT spectrum")
! 466             except Exception as e:
! 467                 if verbose:
! 468                     print(f"  ⚠ FFT plot failed: {e}")
  469 
  470             try:
  471                 plots["histogram"] = plot_histogram(trace.data)
  472                 if verbose:
! 473                     print("  ✓ Generated histogram")
! 474             except Exception as e:
! 475                 if verbose:
! 476                     print(f"  ⚠ Histogram plot failed: {e}")
  477 
  478             try:
  479                 plots["spectrogram"] = plot_spectrogram(trace)
  480                 if verbose:
! 481                     print("  ✓ Generated spectrogram")
! 482             except Exception as e:
! 483                 if verbose:
! 484                     print(f"  ⚠ Spectrogram plot failed: {e}")
  485 
  486             try:
  487                 plots["statistics"] = plot_statistics_summary(trace.data)
! 488                 if verbose:
! 489                     print("  ✓ Generated statistics summary")
  490             except Exception as e:
  491                 if verbose:
! 492                     print(f"  ⚠ Statistics plot failed: {e}")
  493     else:
  494         # Digital signal plots
  495         from oscura.core.types import DigitalTrace as DigitalTraceType
```


---


Lines 497-508

```python
  497         if isinstance(trace, DigitalTraceType):  # Type narrowing
  498             try:
  499                 plots["logic"] = plot_logic_analyzer(trace)
  500                 if verbose:
! 501                     print("  ✓ Generated logic analyzer view")
! 502             except Exception as e:
! 503                 if verbose:
! 504                     print(f"  ⚠ Logic analyzer plot failed: {e}")
  505 
  506     return plots
  507 
```


---


## src/oscura/workflows/waveform&#46;py

Lines 121-306

```python
  121     if not filepath.exists():
  122         raise FileNotFoundError(f"File not found: {filepath}")
  123 
  124     # Set up output directory
! 125     if output_dir is None:
! 126         output_dir = Path("./waveform_analysis_output")
  127     else:
! 128         output_dir = Path(output_dir)
! 129     output_dir.mkdir(parents=True, exist_ok=True)
  130 
  131     # Determine which analyses to run
! 132     valid_analyses = {"time_domain", "frequency_domain", "digital", "statistics"}
! 133     if analyses == "all":
! 134         requested_analyses = list(valid_analyses)
  135     else:
! 136         requested_analyses = analyses
! 137         invalid = set(requested_analyses) - valid_analyses
! 138         if invalid:
! 139             raise ValueError(f"Invalid analysis types: {invalid}. Valid: {valid_analyses}")
  140 
! 141     if verbose:
! 142         print("=" * 80)
! 143         print("OSCURA COMPLETE WAVEFORM ANALYSIS WITH REVERSE ENGINEERING")
! 144         print("=" * 80)
! 145         print(f"\nLoading: {filepath.name}")
  146 
  147     # Step 1: Load waveform
! 148     trace = osc.load(filepath)
  149 
  150     # Detect signal type using new properties
! 151     is_digital = (
  152         trace.is_digital if hasattr(trace, "is_digital") else isinstance(trace, DigitalTrace)
  153     )
  154 
! 155     if verbose:
! 156         signal_type = "Digital" if is_digital else "Analog"
! 157         print(f"✓ Loaded {signal_type} signal")
! 158         print(f"  Samples: {len(trace)}")
! 159         print(f"  Sample rate: {trace.metadata.sample_rate:.2e} Hz")
! 160         print(f"  Duration: {trace.duration:.6f} s")
  161 
  162     # Step 2: Run basic analyses
! 163     results: dict[str, dict[str, Any]] = {}
  164 
! 165     if "time_domain" in requested_analyses:
! 166         if verbose:
! 167             print("\n" + "=" * 80)
! 168             print("TIME-DOMAIN ANALYSIS")
! 169             print("=" * 80)
  170 
  171         # Run time-domain measurements - GET ALL AVAILABLE
! 172         if isinstance(trace, WaveformTrace):
! 173             from oscura.analyzers import waveform as waveform_analyzer
  174 
  175             # Pass parameters=None to get ALL available measurements
! 176             time_results = waveform_analyzer.measure(trace, parameters=None, include_units=True)
! 177             results["time_domain"] = time_results
! 178             if verbose:
! 179                 print(f"✓ Completed {len(time_results)} measurements")
  180 
! 181     if "frequency_domain" in requested_analyses and not is_digital:
! 182         if verbose:
! 183             print("\n" + "=" * 80)
! 184             print("FREQUENCY-DOMAIN ANALYSIS")
! 185             print("=" * 80)
  186 
! 187         if isinstance(trace, WaveformTrace):
  188             # Run spectral analysis using unified measure() API
! 189             from oscura.analyzers.waveform import spectral
  190 
! 191             freq_results = spectral.measure(trace, include_units=True)
  192 
  193             # Add FFT arrays for plotting (not measurements)
! 194             try:
! 195                 fft_result = osc.fft(trace)
! 196                 freq_results["fft_freqs"] = fft_result[0]
! 197                 freq_results["fft_data"] = fft_result[1]
! 198             except Exception:
! 199                 pass
  200 
! 201             results["frequency_domain"] = freq_results
! 202             if verbose:
  203                 # Count actual measurements (not arrays)
! 204                 numeric_count = sum(
  205                     1
  206                     for k, v in freq_results.items()
  207                     if k not in ["fft_freqs", "fft_data"] and isinstance(v, (int, float, dict))
  208                 )
! 209                 print(f"✓ Completed {numeric_count} measurements")
  210 
! 211     if "digital" in requested_analyses:
! 212         if verbose:
! 213             print("\n" + "=" * 80)
! 214             print("DIGITAL SIGNAL ANALYSIS")
! 215             print("=" * 80)
  216 
  217         # Run comprehensive digital analysis (works for both analog and digital traces)
! 218         try:
! 219             from oscura.analyzers.digital import signal_quality_summary, timing
  220 
  221             # Convert DigitalTrace to WaveformTrace for analysis
! 222             analysis_trace = trace
! 223             if isinstance(trace, DigitalTrace):
  224                 # Convert bool array to float for analysis
! 225                 waveform_data = trace.data.astype(float)
! 226                 analysis_trace = WaveformTrace(data=waveform_data, metadata=trace.metadata)
  227 
! 228             if isinstance(analysis_trace, WaveformTrace):
  229                 # Get signal quality summary
! 230                 digital_results_obj = signal_quality_summary(analysis_trace)
  231                 digital_results: dict[str, Any]
! 232                 if hasattr(digital_results_obj, "__dict__"):
! 233                     digital_results = digital_results_obj.__dict__
  234                 else:
! 235                     digital_results = dict(digital_results_obj)
  236 
  237                 # Add timing measurements
! 238                 try:
  239                     # Slew rate for rising and falling edges
! 240                     slew_rising = timing.slew_rate(
  241                         analysis_trace, edge_type="rising", return_all=False
  242                     )
! 243                     if not np.isnan(slew_rising):
! 244                         digital_results["slew_rate_rising"] = slew_rising
  245 
! 246                     slew_falling = timing.slew_rate(
  247                         analysis_trace, edge_type="falling", return_all=False
  248                     )
! 249                     if not np.isnan(slew_falling):
! 250                         digital_results["slew_rate_falling"] = slew_falling
! 251                 except Exception:
! 252                     pass  # Skip if slew rate not applicable
  253 
! 254                 results["digital"] = digital_results
! 255                 if verbose:
! 256                     numeric_count = sum(
  257                         1 for v in digital_results.values() if isinstance(v, (int, float))
  258                     )
! 259                     print(f"✓ Completed {numeric_count} measurements")
! 260         except Exception as e:
! 261             if verbose:
! 262                 print(f"  ⚠ Digital analysis unavailable: {e}")
  263 
! 264     if "statistics" in requested_analyses and not is_digital:
! 265         if verbose:
! 266             print("\n" + "=" * 80)
! 267             print("STATISTICAL ANALYSIS")
! 268             print("=" * 80)
  269 
! 270         if isinstance(trace, WaveformTrace):
  271             # Run statistical analysis using unified measure() API
! 272             from oscura.analyzers import statistics
  273 
! 274             stats_results = statistics.measure(trace.data, include_units=True)
! 275             results["statistics"] = stats_results
! 276             if verbose:
! 277                 numeric_count = len(stats_results)
! 278                 print(f"✓ Completed {numeric_count} measurements")
  279 
  280     # Step 3: Protocol Detection & Decoding (FULL IMPLEMENTATION)
! 281     protocols_detected: list[dict[str, Any]] = []
! 282     decoded_frames: list[Any] = []
  283 
! 284     if enable_protocol_decode and is_digital:
! 285         if verbose:
! 286             print("\n" + "=" * 80)
! 287             print("PROTOCOL DETECTION & DECODING")
! 288             print("=" * 80)
  289 
! 290         try:
! 291             from oscura.discovery import auto_decoder
  292 
  293             # Try each protocol hint or auto-detect
! 294             protocols_to_try = protocol_hints if protocol_hints else ["UART", "SPI", "I2C"]
  295 
! 296             for proto_name in protocols_to_try:
! 297                 try:
  298                     # Type narrow to WaveformTrace | DigitalTrace
! 299                     if not isinstance(trace, (WaveformTrace, DigitalTrace)):
! 300                         continue
  301 
! 302                     result = auto_decoder.decode_protocol(
  303                         trace,
  304                         protocol_hint=proto_name.upper(),  # type: ignore[arg-type]
  305                         confidence_threshold=0.6,  # Lower threshold to catch more
  306                     )
```


---


Lines 304-313

```python
  304                         protocol_hint=proto_name.upper(),  # type: ignore[arg-type]
  305                         confidence_threshold=0.6,  # Lower threshold to catch more
  306                     )
  307 
! 308                     if result.overall_confidence >= 0.6:
! 309                         proto_info = {
  310                             "protocol": result.protocol,
  311                             "confidence": result.overall_confidence,
  312                             "params": result.detected_params,
  313                             "frame_count": result.frame_count,
```


---


Lines 312-345

```python
  312                             "params": result.detected_params,
  313                             "frame_count": result.frame_count,
  314                             "error_count": result.error_count,
  315                         }
! 316                         protocols_detected.append(proto_info)
! 317                         decoded_frames.extend(result.data)
  318 
! 319                         if verbose:
! 320                             print(
  321                                 f"✓ Detected {result.protocol.upper()}: "
  322                                 f"{result.overall_confidence:.1%} confidence"
  323                             )
! 324                             print(
  325                                 f"  Decoded {len(result.data)} bytes, {result.frame_count} frames"
  326                             )
! 327                 except Exception:
  328                     # Protocol didn't match, continue trying others
! 329                     pass
  330 
! 331             if not protocols_detected and verbose:
! 332                 print("  ⚠ No protocols detected (signal may be unknown or noisy)")
  333 
! 334         except Exception as e:
! 335             if verbose:
! 336                 print(f"  ⚠ Protocol detection unavailable: {e}")
  337 
  338     # Step 4: Reverse Engineering Pipeline (FULL IMPLEMENTATION)
! 339     reverse_engineering_results: dict[str, Any] | None = None
  340 
! 341     if (
  342         enable_reverse_engineering
  343         and is_digital
  344         and isinstance(trace, (WaveformTrace, DigitalTrace))
  345         and len(trace.data) > 1000
```


---


Lines 343-383

```python
  343         and is_digital
  344         and isinstance(trace, (WaveformTrace, DigitalTrace))
  345         and len(trace.data) > 1000
  346     ):
! 347         if verbose:
! 348             print("\n" + "=" * 80)
! 349             print("REVERSE ENGINEERING ANALYSIS")
! 350             print("=" * 80)
! 351             depth_map = {
  352                 "quick": "Quick (basic)",
  353                 "standard": "Standard (comprehensive)",
  354                 "deep": "Deep (exhaustive)",
  355             }
! 356             print(f"  Mode: {depth_map.get(reverse_engineering_depth, 'Standard')}")
  357 
! 358         try:
! 359             from oscura.workflows import reverse_engineering as re_workflow
  360 
  361             # Convert DigitalTrace to WaveformTrace for RE
! 362             re_trace = trace
! 363             if isinstance(trace, DigitalTrace):
! 364                 waveform_data = trace.data.astype(float)
! 365                 re_trace = WaveformTrace(data=waveform_data, metadata=trace.metadata)
  366 
! 367             if isinstance(re_trace, WaveformTrace):
  368                 # Set parameters based on depth
! 369                 if reverse_engineering_depth == "quick":
! 370                     baud_rates = [9600, 115200]
! 371                     min_frames = 2
! 372                 elif reverse_engineering_depth == "deep":
! 373                     baud_rates = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
! 374                     min_frames = 5
  375                 else:  # standard
! 376                     baud_rates = [9600, 19200, 38400, 57600, 115200, 230400]
! 377                     min_frames = 3
  378 
! 379                 re_result = re_workflow.reverse_engineer_signal(
  380                     re_trace,
  381                     expected_baud_rates=baud_rates,
  382                     min_frames=min_frames,
  383                     max_frame_length=256,
```


---


Lines 383-391

```python
  383                     max_frame_length=256,
  384                 )
  385 
  386                 # Extract key findings
! 387                 reverse_engineering_results = {
  388                     "baud_rate": re_result.baud_rate,
  389                     "confidence": re_result.confidence,
  390                     "frame_count": len(re_result.frames),
  391                     "frame_format": re_result.protocol_spec.frame_format,
```


---


Lines 396-450

```python
  396                     "checksum_position": re_result.protocol_spec.checksum_position,
  397                     "warnings": re_result.warnings,
  398                 }
  399 
! 400                 if verbose:
! 401                     print(
  402                         f"✓ Baud rate: {re_result.baud_rate:.0f} Hz (confidence: {re_result.confidence:.1%})"
  403                     )
! 404                     print(f"✓ Frames: {len(re_result.frames)} detected")
! 405                     if re_result.protocol_spec.sync_pattern:
! 406                         print(f"✓ Sync pattern: {re_result.protocol_spec.sync_pattern}")
! 407                     if re_result.protocol_spec.frame_length:
! 408                         print(f"✓ Frame length: {re_result.protocol_spec.frame_length} bytes")
! 409                     if re_result.protocol_spec.checksum_type:
! 410                         print(f"✓ Checksum: {re_result.protocol_spec.checksum_type}")
! 411                     if re_result.warnings:
! 412                         print(f"  ⚠ Warnings: {len(re_result.warnings)}")
  413 
! 414         except ValueError as e:
! 415             if verbose:
! 416                 print(f"  ⚠ RE analysis: {e!s}")
! 417             reverse_engineering_results = {"status": "insufficient_data", "message": str(e)}
! 418         except Exception as e:
! 419             if verbose:
! 420                 print(f"  ⚠ RE analysis unavailable: {e}")
! 421             reverse_engineering_results = {"status": "error", "message": str(e)}
  422 
  423     # Step 5: Pattern Recognition & Anomaly Detection (FULL IMPLEMENTATION)
! 424     pattern_results: dict[str, Any] | None = None
! 425     anomalies_detected: list[dict[str, Any]] = []
  426 
! 427     if enable_pattern_recognition:
! 428         if verbose:
! 429             print("\n" + "=" * 80)
! 430             print("PATTERN RECOGNITION & ANOMALY DETECTION")
! 431             print("=" * 80)
  432 
! 433         pattern_results = {}
  434 
  435         # Anomaly detection
! 436         try:
! 437             from oscura.discovery import anomaly_detector
  438 
  439             # Convert DigitalTrace to WaveformTrace for anomaly detection
! 440             anomaly_trace = trace
! 441             if isinstance(trace, DigitalTrace):
! 442                 waveform_data = trace.data.astype(float)
! 443                 anomaly_trace = WaveformTrace(data=waveform_data, metadata=trace.metadata)
  444 
! 445             if isinstance(anomaly_trace, WaveformTrace):
! 446                 anomalies = anomaly_detector.find_anomalies(
  447                     anomaly_trace,
  448                     min_confidence=0.6,
  449                 )
```


---


Lines 448-456

```python
  448                     min_confidence=0.6,
  449                 )
  450 
  451                 # Convert to list of dicts
! 452                 anomalies_detected = [
  453                     {
  454                         "type": a.type,
  455                         "start": float(a.timestamp_us) / 1e6,  # Convert to seconds
  456                         "end": float(a.timestamp_us + a.duration_ns / 1000) / 1e6,
```


---


Lines 458-488

```python
  458                         "description": a.description,
  459                     }
  460                     for a in anomalies
  461                 ]
! 462                 pattern_results["anomalies"] = anomalies_detected
  463 
! 464                 if verbose and anomalies_detected:
! 465                     print(f"✓ Detected {len(anomalies_detected)} anomalies")
! 466                     severity_counts: dict[str, int] = {}
! 467                     for a in anomalies_detected:
! 468                         severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1
! 469                     for sev, count in sorted(severity_counts.items()):
! 470                         print(f"  - {sev}: {count}")
! 471         except Exception as e:
! 472             if verbose:
! 473                 print(f"  ⚠ Anomaly detection unavailable: {e}")
  474 
  475         # Pattern discovery (for byte streams)
! 476         if decoded_frames and len(decoded_frames) > 10:
! 477             try:
! 478                 from oscura.analyzers.patterns import discovery
  479 
  480                 # Convert decoded bytes to numpy array
! 481                 byte_data = np.array([b.value for b in decoded_frames[:1000]], dtype=np.uint8)
  482 
! 483                 signatures = discovery.discover_signatures(byte_data, min_occurrences=3)
! 484                 pattern_results["signatures"] = [
  485                     {
  486                         "pattern": sig.pattern.hex(),
  487                         "count": sig.occurrences,
  488                         "confidence": float(sig.score),
```


---


Lines 490-535

```python
  490                     }
  491                     for sig in signatures[:10]  # Top 10
  492                 ]
  493 
! 494                 if verbose and signatures:
! 495                     print(f"✓ Discovered {len(signatures)} signature patterns")
! 496             except Exception as e:
! 497                 if verbose:
! 498                     print(f"  ⚠ Pattern discovery unavailable: {e}")
  499 
  500     # Step 6: Generate plots (ALL plots from original + RE plots)
! 501     plots: dict[str, str] = {}
! 502     if generate_plots:
! 503         if verbose:
! 504             print("\n" + "=" * 80)
! 505             print("GENERATING VISUALIZATIONS")
! 506             print("=" * 80)
  507 
! 508         from oscura.visualization import batch
  509 
  510         # Generate ALL standard plots
! 511         if isinstance(trace, (WaveformTrace, DigitalTrace)):
! 512             plots = batch.generate_all_plots(trace, verbose=verbose)
  513 
! 514         if verbose:
! 515             print(f"✓ Generated {len(plots)} total plots")
  516 
  517     # Step 7: Generate comprehensive report
! 518     report_path: Path | None = None
! 519     if generate_report:
! 520         if verbose:
! 521             print("\n" + "=" * 80)
! 522             print("GENERATING COMPREHENSIVE REPORT")
! 523             print("=" * 80)
  524 
! 525         from oscura.reporting import Report, ReportConfig, generate_html_report
  526 
  527         # Create report
! 528         valid_format: Literal["html", "pdf", "markdown", "docx"] = (
  529             "html" if report_format == "html" else "pdf"
  530         )
! 531         config = ReportConfig(
  532             title="Complete Waveform Analysis with Reverse Engineering",
  533             format=valid_format,
  534             verbosity="detailed",
  535         )
```


---


Lines 533-541

```python
  533             format=valid_format,
  534             verbosity="detailed",
  535         )
  536 
! 537         report = Report(
  538             config=config,
  539             metadata={
  540                 "file": str(filepath),
  541                 "type": "Digital" if is_digital else "Analog",
```


---


Lines 545-569

```python
  545             },
  546         )
  547 
  548         # Add basic measurement sections - handle BOTH formats
! 549         for analysis_name, analysis_results in results.items():
  550             # Extract measurements in both formats:
  551             # 1. Unified format: {"value": float, "unit": str}
  552             # 2. Legacy format: flat float/int values
! 553             measurements = {}
  554 
! 555             for k, v in analysis_results.items():
! 556                 if isinstance(v, dict) and "value" in v:
  557                     # Unified format - extract value for reporting
! 558                     measurements[k] = v["value"]
! 559                 elif isinstance(v, (int, float)) and not isinstance(v, bool):
  560                     # Legacy flat format
! 561                     measurements[k] = v
  562                 # Skip arrays, objects, etc.
  563 
! 564             if measurements:
! 565                 title_map = {
  566                     "time_domain": "Time-Domain Analysis (IEEE 181-2011)",
  567                     "frequency_domain": "Frequency-Domain Analysis (IEEE 1241-2010)",
  568                     "digital": "Digital Signal Analysis",
  569                     "statistics": "Statistical Analysis",
```


---


Lines 567-580

```python
  567                     "frequency_domain": "Frequency-Domain Analysis (IEEE 1241-2010)",
  568                     "digital": "Digital Signal Analysis",
  569                     "statistics": "Statistical Analysis",
  570                 }
! 571                 title = title_map.get(analysis_name, analysis_name.replace("_", " ").title())
! 572                 report.add_measurements(title=title, measurements=measurements)
  573 
  574         # Add protocol detection section
! 575         if protocols_detected:
! 576             report.add_section(
  577                 title="Protocol Detection Results",
  578                 content=_format_protocol_detection(protocols_detected, decoded_frames),
  579             )
```


---


Lines 578-587

```python
  578                 content=_format_protocol_detection(protocols_detected, decoded_frames),
  579             )
  580 
  581         # Add reverse engineering section
! 582         if reverse_engineering_results and reverse_engineering_results.get("baud_rate"):
! 583             report.add_section(
  584                 title="Reverse Engineering Analysis",
  585                 content=_format_reverse_engineering(reverse_engineering_results),
  586             )
```


---


Lines 585-594

```python
  585                 content=_format_reverse_engineering(reverse_engineering_results),
  586             )
  587 
  588         # Add anomaly detection section
! 589         if anomalies_detected:
! 590             report.add_section(
  591                 title="Anomaly Detection Results",
  592                 content=_format_anomalies(anomalies_detected),
  593             )
```


---


Lines 592-601

```python
  592                 content=_format_anomalies(anomalies_detected),
  593             )
  594 
  595         # Add pattern recognition section
! 596         if pattern_results and pattern_results.get("signatures"):
! 597             report.add_section(
  598                 title="Pattern Recognition Results",
  599                 content=_format_patterns(pattern_results),
  600             )
```


---


Lines 599-637

```python
  599                 content=_format_patterns(pattern_results),
  600             )
  601 
  602         # Generate HTML
! 603         html_content = generate_html_report(report)
  604 
  605         # Embed plots if requested
! 606         if embed_plots and plots:
! 607             from oscura.reporting import embed_plots as embed_plots_func
  608 
! 609             html_content = embed_plots_func(html_content, plots)
! 610             if verbose:
! 611                 print(f"  ✓ Embedded {len(plots)} plots in report")
  612 
  613         # Save report
! 614         report_path = output_dir / f"complete_analysis.{report_format}"
! 615         report_path.write_text(html_content, encoding="utf-8")
  616 
! 617         if verbose:
! 618             print(f"✓ Report saved: {report_path}")
  619 
! 620     if verbose:
! 621         print("\n" + "=" * 80)
! 622         print("ANALYSIS COMPLETE")
! 623         print("=" * 80)
! 624         print(f"✓ Output directory: {output_dir}")
! 625         if protocols_detected:
! 626             print(f"✓ Protocols detected: {len(protocols_detected)}")
! 627         if decoded_frames:
! 628             print(f"✓ Frames decoded: {len(decoded_frames)}")
! 629         if anomalies_detected:
! 630             print(f"✓ Anomalies found: {len(anomalies_detected)}")
  631 
  632     # Return comprehensive results
! 633     return {
  634         "filepath": filepath,
  635         "trace": trace,
  636         "is_digital": is_digital,
  637         "results": results,
```


---


Lines 655-677

```python
  655 
  656     Returns:
  657         HTML formatted string.
  658     """
! 659     html = "<h3>Detected Protocols</h3>\n<ul>\n"
! 660     for proto in protocols:
! 661         conf = proto.get("confidence", 0.0)
! 662         html += f"<li><strong>{proto['protocol'].upper()}</strong>: {conf:.1%} confidence"
! 663         if "params" in proto and "baud_rate" in proto["params"]:
! 664             html += f" at {proto['params']['baud_rate']:.0f} baud"
! 665         if proto.get("frame_count"):
! 666             html += f" ({proto['frame_count']} frames)"
! 667         html += "</li>\n"
! 668     html += "</ul>\n"
  669 
! 670     if frames:
! 671         html += f"<p><strong>Total bytes decoded:</strong> {len(frames)}</p>\n"
  672 
! 673     return html
  674 
  675 
  676 def _format_reverse_engineering(re_results: dict[str, Any]) -> str:
  677     """Format reverse engineering results for report.
```


---


Lines 681-727

```python
  681 
  682     Returns:
  683         HTML formatted string.
  684     """
! 685     html = "<h3>Reverse Engineering Findings</h3>\n<ul>\n"
  686 
! 687     if re_results.get("baud_rate"):
! 688         html += f"<li><strong>Baud Rate:</strong> {re_results['baud_rate']:.0f} Hz</li>\n"
  689 
! 690     if re_results.get("confidence"):
! 691         conf = re_results["confidence"]
! 692         html += f"<li><strong>Overall Confidence:</strong> {conf:.1%}</li>\n"
  693 
! 694     if re_results.get("frame_count"):
! 695         html += f"<li><strong>Frames Detected:</strong> {re_results['frame_count']}</li>\n"
  696 
! 697     if re_results.get("frame_format"):
! 698         html += f"<li><strong>Frame Format:</strong> {re_results['frame_format']}</li>\n"
  699 
! 700     if re_results.get("sync_pattern"):
! 701         html += f"<li><strong>Sync Pattern:</strong> {re_results['sync_pattern']}</li>\n"
  702 
! 703     if re_results.get("frame_length"):
! 704         html += f"<li><strong>Frame Length:</strong> {re_results['frame_length']} bytes</li>\n"
  705 
! 706     if re_results.get("field_count"):
! 707         html += f"<li><strong>Fields Identified:</strong> {re_results['field_count']}</li>\n"
  708 
! 709     if re_results.get("checksum_type"):
! 710         html += f"<li><strong>Checksum:</strong> {re_results['checksum_type']}"
! 711         if re_results.get("checksum_position") is not None:
! 712             html += f" at position {re_results['checksum_position']}"
! 713         html += "</li>\n"
  714 
! 715     html += "</ul>\n"
  716 
! 717     if re_results.get("warnings"):
! 718         html += "<h4>Warnings</h4>\n<ul>\n"
! 719         for warning in re_results["warnings"][:5]:  # Max 5 warnings
! 720             html += f"<li>{warning}</li>\n"
! 721         html += "</ul>\n"
  722 
! 723     return html
  724 
  725 
  726 def _format_anomalies(anomalies: list[dict[str, Any]]) -> str:
  727     """Format anomaly detection results for report.
```


---


Lines 731-756

```python
  731 
  732     Returns:
  733         HTML formatted string.
  734     """
! 735     html = "<h3>Detected Anomalies</h3>\n"
! 736     html += f"<p><strong>Total anomalies:</strong> {len(anomalies)}</p>\n"
  737 
  738     # Group by severity
! 739     by_severity: dict[str, list[dict[str, Any]]] = {}
! 740     for anomaly in anomalies:
! 741         severity = anomaly.get("severity", "unknown")
! 742         by_severity.setdefault(severity, []).append(anomaly)
  743 
! 744     for severity in ["critical", "warning", "info"]:
! 745         if severity in by_severity:
! 746             html += f"<h4>{severity.title()} ({len(by_severity[severity])})</h4>\n<ul>\n"
! 747             for anomaly in by_severity[severity][:10]:  # Max 10 per severity
! 748                 html += f"<li><strong>{anomaly['type']}:</strong> {anomaly['description']}"
! 749                 html += f" (at {anomaly['start']:.6f}s)</li>\n"
! 750             html += "</ul>\n"
  751 
! 752     return html
  753 
  754 
  755 def _format_patterns(pattern_results: dict[str, Any]) -> str:
  756     """Format pattern recognition results for report.
```


---


Lines 760-782

```python
  760 
  761     Returns:
  762         HTML formatted string.
  763     """
! 764     html = "<h3>Pattern Recognition Results</h3>\n"
  765 
! 766     if pattern_results.get("signatures"):
! 767         sigs = pattern_results["signatures"]
! 768         html += f"<p><strong>Signature patterns discovered:</strong> {len(sigs)}</p>\n"
! 769         html += "<table border='1' cellpadding='5'>\n"
! 770         html += "<tr><th>Pattern</th><th>Length</th><th>Count</th><th>Score</th></tr>\n"
! 771         for sig in sigs[:10]:  # Top 10
! 772             html += f"<tr><td><code>{sig['pattern']}</code></td>"
! 773             html += f"<td>{sig['length']} bytes</td>"
! 774             html += f"<td>{sig['count']}</td>"
! 775             html += f"<td>{sig['confidence']:.2f}</td></tr>\n"
! 776         html += "</table>\n"
  777 
! 778     return html
  779 
  780 
  781 __all__ = [
  782     "analyze_complete",
```


---

