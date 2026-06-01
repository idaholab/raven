
# Optimization Plot Catalogue

This catalogue enumerates the optimizer-oriented `Plot` OutStreams that ship with RAVEN, indicates which requests from the recent visualization survey are already covered, and captures example workflows pulled directly from the regression suite. Use it as a quick reference when wiring plots into new studies or when deciding where to invest in future visual diagnostics.

## Implemented Plot Interfaces

| Status | Plot Interface | Primary Insight | Implementation | Example Workflow |
| --- | --- | --- | --- | --- |
| [x] | OptPath | Decision-variable trajectory with accept/reject markers | `ravenframework/OutStreams/PlotInterfaces/OptPath.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:160` |
| [x] | PopulationPlot | Min/avg/max envelopes per generation | `ravenframework/OutStreams/PlotInterfaces/PopulationPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskContinuestrained.xml:165` |
| [x] | OptParallelCoordinatePlot | Parallel coordinate beams across generations (optional constraint-encoded color/width) | `ravenframework/OutStreams/PlotInterfaces/OptParallelCoordinate.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:171` |
| [x] | ObjectiveContourAnimationPlot | Animated contour map with feasibility overlays | `ravenframework/OutStreams/PlotInterfaces/ObjectiveContourAnimation.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:176` |
| [x] | FeasibleRegionObjectiveContourPlot | Decision-space feasible region + objective/constraint contours (3D panels) | `ravenframework/OutStreams/PlotInterfaces/FeasibleRegionObjectiveContourPlot.py` | `plugins/PRLO/examples/AP1000/test_ap1000_multiobjective.xml:730` |
| [x] | ConstraintActivityTimelinePlot | Per-constraint violation timeline | `ravenframework/OutStreams/PlotInterfaces/ConstraintActivityTimelinePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:187` |
| [x] | FitnessFunnelPlot | Best/mean fitness plus variance band | `ravenframework/OutStreams/PlotInterfaces/FitnessFunnelPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:196` |
| [x] | SamplingCoverageMapPlot | Animated 2D sampling density + scatter | `ravenframework/OutStreams/PlotInterfaces/SamplingCoverageMapPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml:202` |
| [x] | NSGAParetoFrontPlot | Static Rank-1 Pareto front (2D/3D) | `ravenframework/OutStreams/PlotInterfaces/NSGAParetoFrontPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:205` |
| [x] | DecisionObjectiveMappingPlot | Side-by-side decision/objective space mapping (optional linking lines) | `ravenframework/OutStreams/PlotInterfaces/DecisionObjectiveMappingPlot.py` | (example TBD) |
| [x] | NSGAFrontAnimation | Pareto front evolution animation | `ravenframework/OutStreams/PlotInterfaces/NSGAFrontAnimation.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:210` |
| [x] | NSGAFrontRankAnimation | Rank-colored front animation | `ravenframework/OutStreams/PlotInterfaces/NSGAFrontRankAnimation.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:229` |
| [x] | NSGAIIIReferenceDirectionPlot | NSGA-III reference direction coverage on the simplex | `ravenframework/OutStreams/PlotInterfaces/NSGAIIIReferenceDirectionPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:187` |
| [x] | NSGAIIINichingHeatmapPlot | Reference-direction niching occupancy heatmap | `ravenframework/OutStreams/PlotInterfaces/NSGAIIINichingHeatmapPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:195` |
| [x] | HypervolumeMoviePlot | Hypervolume progression over generations | `ravenframework/OutStreams/PlotInterfaces/HypervolumeMoviePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:253` |
| [x] | TradeoffSlicePlot | Animated pairwise trade-off density slices | `ravenframework/OutStreams/PlotInterfaces/TradeoffSlicePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:247` |
| [x] | BubbleTradeoffPlot | Bubble chart (2D) or 3D scatter with bubble size encoding a fourth metric | `ravenframework/OutStreams/PlotInterfaces/BubbleTradeoffPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:212` |
| [x] | ThreeDVectorPlot | Arrow field showing generation-to-generation drift in objective space | `ravenframework/OutStreams/PlotInterfaces/ThreeDVectorPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:214` |
| [x] | ThreeDTubePlot | Thick 3D trajectory of the best sample per generation | `ravenframework/OutStreams/PlotInterfaces/ThreeDTubePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:220` |
| [x] | ParetoSurfacePlot | Triangulated rank-1 surface approximation for three objectives | `ravenframework/OutStreams/PlotInterfaces/ParetoSurfacePlot.py` | (example TBD) |
| [x] | ThreeDConePlot | Dominance cones emanating from the utopia point toward elite samples | `ravenframework/OutStreams/PlotInterfaces/ThreeDConePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/DTLZ1_NSGAIII.xml:226` |
| [x] | DominanceHeatMapPlot | Dominated vs non-dominated density heatmap | `ravenframework/OutStreams/PlotInterfaces/DominanceHeatMapPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:257` |
| [x] | ParetoDiagnosticsPlot | Hypervolume + rank-1 count diagnostics | `ravenframework/OutStreams/PlotInterfaces/ParetoDiagnosticsPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:240` |
| [x] | AttainmentSurfacePlot | Empirical attainment probability contours | `ravenframework/OutStreams/PlotInterfaces/AttainmentSurfacePlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:266` |
| [x] | ResponseSurfaceOverlayPlot | Contoured response surface with sample overlay | `ravenframework/OutStreams/PlotInterfaces/ResponseSurfaceOverlayPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:273` |
| [x] | MultiRunUncertaintyPlot | Mean/quantile bands across runs | `ravenframework/OutStreams/PlotInterfaces/MultiRunUncertaintyPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:279` |
| [x] | RadvizEmbeddingPlot | Radial embedding of many decision variables | `ravenframework/OutStreams/PlotInterfaces/RadvizEmbeddingPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:331` |
| [x] | ProsectionMatrixPlot | Pairwise slices near high-dimensional medians | `ravenframework/OutStreams/PlotInterfaces/ProsectionMatrixPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:337` |
| [x] | PreferenceSweepAnimationPlot | Weighted-preference sweep across objectives | `ravenframework/OutStreams/PlotInterfaces/PreferenceSweepAnimationPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:344` |
| [x] | StarCoordinatesPlot | Star-coordinate embedding of samples | `ravenframework/OutStreams/PlotInterfaces/StarCoordinatesPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:355` |
| [x] | SelfOrganizingMapPlot | SOM lattice occupancy and colour map | `ravenframework/OutStreams/PlotInterfaces/SelfOrganizingMapPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:361` |
| [x] | ChordDiagramPlot | Circular correlation chord diagram | `ravenframework/OutStreams/PlotInterfaces/ChordDiagramPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:369` |
| [x] | GlyphRadarPlot | Radar glyphs for representative samples | `ravenframework/OutStreams/PlotInterfaces/GlyphRadarPlot.py` | `tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml:376` |
| [x] | FeasibilityRadarPlot | Radar/spider summary comparing feasible vs infeasible samples (optional quantile bands) | `ravenframework/OutStreams/PlotInterfaces/FeasibilityRadarPlot.py` | (example TBD) |
| [x] | ParetoChartPlot | Pareto chart (sorted bars + cumulative %) for categories or constraint violations | `ravenframework/OutStreams/PlotInterfaces/ParetoChartPlot.py` | (example TBD) |
| [x] | AdjustedEpsilonOptimalPlot | Epsilon-grid view and epsilon-efficient representatives in 2D objective space (supports `axis_space=objective`) | `ravenframework/OutStreams/PlotInterfaces/AdjustedEpsilonOptimalPlot.py` | (example TBD) |
| [ ] | DiversityRadarPlot | Crowding-distance radar animation (needs regression harness) | `ravenframework/OutStreams/PlotInterfaces/DiversityRadarPlot.py` | _Example XML pending (plot is available but not yet wired into a regression test)_ |

## Wishlist: 3D/High-Dimensional Concepts Not Yet Implemented

| Concept | Motivation | Notes |
| --- | --- | --- |
| 3D Density / Isosurface Plot | Volume rendering of attainment probability | Present AttainmentSurface is 2D; true 3D needs iso-surface computations. |
| 3D Voxel / Volume Rendering | Occupancy heatmap through the volume | No voxel-based OutStream yet. |
| 3D Contour Plot | Constant-value surfaces for three objectives | ObjectiveContourAnimation only supports 2D slices. |
| 3D Kernel Density Estimation | Smooth 3D PDF of Pareto samples | Would extend DominanceHeatMap beyond 2D. |
| 3D PCA Projection | Project many objectives down to 3D | No PCA-based visual exists; could feed interactive scatter or animation. |
| 3D t-SNE / UMAP | Nonlinear embedding of high-dimensional tradeoffs | Manifold-learning visuals are absent. |
| 3D Reference-Point Plot | Plot reference weights directly in 3D space | NSGAIIIReferenceDirectionPlot projects onto a 2D simplex only. |
| 3D Animation Framework | Camera fly-bys / rotating panoramas | Current animations have fixed viewpoints. |
| 3D Rotating Plane | Sweep a slicing plane through objective space | Would extend TradeoffSlicePlot into 3D. |
| 3D Dominance Cone | Visualise dominance regions per solution | Not supported yet. |
| 3D Attainment Surface | True 3D probability fronts | Requires volumetric attainment calculations. |
| 3D Hypervolume Contribution | Visualise contribution per point | Today hypervolume is scalar-only. |
| 3D Decision-Space Mapping | Project decision variables into 3D linked to objectives | Could combine PCA/UMAP with decision annotations. |

## Example Snippets from the Test Suite

### Single-objective (Rosenbrock disk) diagnostics

These plots ship with the constrained Rosenbrock GA regression and illustrate the single-objective toolchain

```xml
<!-- tests/framework/Optimizers/GeneticAlgorithms/continuous/constrained/testGARosenbrockDiskConstrained.xml -->
<Plot name="opt_path" subType="OptPath">
  <source>opt_export</source>
  <vars>x,y,ans</vars>
</Plot>
<Plot name="population" subType="PopulationPlot">
  <source>opt_export</source>
  <vars>x,y,ans</vars>
  <index>batchId</index>
  <how>png</how>
</Plot>
<Plot name="parallel_coordinates" subType="OptParallelCoordinatePlot">
  <source>opt_export</source>
  <vars>x,y</vars>
  <index>batchId</index>
</Plot>
<Plot name="objective_contour_animation" subType="ObjectiveContourAnimationPlot">
  <source>opt_export</source>
  <axes>x, y</axes>
  <objective>ans</objective>
  <index>batchId</index>
  <constraints>ConstraintEvaluation_constraint1</constraints>
  <format>both</format>
  <save_frames>true</save_frames>
  <frames_max>10</frames_max>
</Plot>
<Plot name="constraint_activity" subType="ConstraintActivityTimelinePlot">
  <source>opt_export</source>
  <constraints>ConstraintEvaluation_constraint1</constraints>
  <index>batchId</index>
  <format>both</format>
</Plot>
<Plot name="fitness_funnel" subType="FitnessFunnelPlot">
  <source>opt_export</source>
  <metric>fitness</metric>
  <index>batchId</index>
  <goal>min</goal>
</Plot>
<Plot name="sampling_coverage" subType="SamplingCoverageMapPlot">
  <source>opt_export</source>
  <variables>x, y</variables>
  <index>batchId</index>
  <format>both</format>
</Plot>
```

### Multi-objective (ZDT1) diagnostics

The NSGA-II ZDT1 regression bundles most of the multi-objective visual toolkit in one IOStep.


```xml
<!-- tests/framework/Optimizers/GeneticAlgorithms/continuous/unconstrained/ZDT1.xml -->
<Plot name="pareto_front" subType="NSGAParetoFrontPlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <color>CD</color>
  <rank>1</rank>
</Plot>
<Plot name="front_evolution" subType="NSGAFrontAnimation">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
  <rank>1</rank>
  <format>html</format>
</Plot>
<Plot name="front_rank_animation" subType="NSGAFrontRankAnimation">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
  <format>html</format>
</Plot>
<Plot name="tradeoff_slices" subType="TradeoffSlicePlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
</Plot>
<Plot name="hypervolume_movie" subType="HypervolumeMoviePlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
</Plot>
<Plot name="pareto_diagnostics" subType="ParetoDiagnosticsPlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
  <!-- Optional: choose which space the hypervolume is computed in. -->
  <!-- <space>objective</space> -->
  <!-- Optional (objective space): per-objective goal directions (min/max). -->
  <!-- <goals>min, min</goals> -->
  <!-- Alternative: use optimizer fitness columns (FitnessEvaluation_<objective>). -->
  <!-- <space>fitness</space> -->
</Plot>
<Plot name="dominance_heatmap" subType="DominanceHeatMapPlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
  <bins>60</bins>
</Plot>
<Plot name="attainment_surface" subType="AttainmentSurfacePlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <run_id>batchId</run_id>
  <levels>0.5, 0.9</levels>
</Plot>
<Plot name="response_surface_overlay" subType="ResponseSurfaceOverlayPlot">
  <source>opt_export</source>
  <axes>x1, x2</axes>
  <response>obj1</response>
  <index>batchId</index>
</Plot>
<Plot name="multi_run_uncertainty" subType="MultiRunUncertaintyPlot">
  <source>opt_export</source>
  <run_id>trajID</run_id>
  <index>batchId</index>
  <metric>obj1</metric>
  <quantiles>0.1, 0.9</quantiles>
  <goal>min</goal>
</Plot>
<Plot name="radviz_embedding" subType="RadvizEmbeddingPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <label>rank</label>
  <index>batchId</index>
</Plot>
<Plot name="prosection_matrix" subType="ProsectionMatrixPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <color>rank</color>
  <index>batchId</index>
</Plot>
<Plot name="preference_sweep" subType="PreferenceSweepAnimationPlot">
  <source>opt_export</source>
  <objectives>obj1, obj2</objectives>
  <index>batchId</index>
  <rank>1</rank>
  <frames>21</frames>
  <format>both</format>
  <save_frames>true</save_frames>
</Plot>
<Plot name="star_coordinates" subType="StarCoordinatesPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <label>rank</label>
  <index>batchId</index>
</Plot>
<Plot name="som_map" subType="SelfOrganizingMapPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <color>obj2</color>
  <index>batchId</index>
  <grid>12,12</grid>
  <iterations>400</iterations>
  <seed>13</seed>
</Plot>
<Plot name="chord_diagram" subType="ChordDiagramPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <threshold>0.25</threshold>
  <index>batchId</index>
</Plot>
<Plot name="glyph_profiles" subType="GlyphRadarPlot">
  <source>opt_export</source>
  <variables>x1, x2, x3, obj1, obj2</variables>
  <count>6</count>
  <label>trajID</label>
  <metric>obj1</metric>
  <index>batchId</index>
</Plot>
```

## Sample Output Gallery

#### A. Single-objective (Rosenbrock disk)

1. **OptPath** — plots the path that variables took during an optimization, including accepted and rejected runs.  
   _What-if cues:_ Accepted markers stalling while rejections wander suggests tuning mutation size or constraint repair before convergence stalls.  
   ![OptPath (Rosenbrock)](./rosenbrockDiskConstrained-1-opt_path.png)

2. **PopulationPlot** — plots population coordinates in input and output space so generation envelopes are visible.  
   _What-if cues:_ If min/max envelopes collapse early the run may need more exploration; if envelopes flare, consider reducing mutation amplitude.  
   ![PopulationPlot (Rosenbrock)](./rosenbrockDiskConstrained-1-population.png)

3. **OptParallelCoordinatePlot** — plots input coordinates on a parallel coordinate chart across generations.  
   _What-if cues:_ Persistent braiding between axes shows decision-variable coupling; apply preference/constraint tweaks to untangle undesired links.  
   ![OptParallelCoordinatePlot (Rosenbrock)](./parallel_coordinates_1.0.png)

4. **ObjectiveContourAnimationPlot** — animation highlighting optimizer populations against concentric contour levels in objective space.  
   _What-if cues:_ If feasible markers stay outside high-value contours, expand sampling or relax constraints; sharp best-point jumps hint at noisy evaluations.  
   ![ObjectiveContourAnimationPlot (Rosenbrock GIF)](./1-objective_contour_animation.gif)

5. **ConstraintActivityTimelinePlot** — animated constraint violation timelines with one subplot per constraint.  
   _What-if cues:_ Subplots stuck near 100 % violation imply chronic infeasibility (consider softer penalties); late spikes often trace to restart policies or survivor steps.  
   ![ConstraintActivityTimelinePlot (Rosenbrock GIF)](./1-constraint_activity.gif)

6. **ConstraintViolationHeatmapPlot** — heatmap showing average constraint violation magnitude across a 2-D variable grid.  
   _What-if cues:_ Localised hotspots expose troublesome variable combinations; if the entire map glows, rescale constraints or re-seed the population.  
   ![ConstraintViolationHeatmapPlot (Rosenbrock)](./1-constraint_heatmap.png)

7. **FitnessFunnelPlot** — visualises generational convergence via best/mean fitness lines and a variance band.  
   _What-if cues:_ Flat best fitness with wide variance signals stagnation -> boost exploration; variance collapsing while best improves steadily confirms healthy exploitation.  
   ![FitnessFunnelPlot (Rosenbrock)](./1-fitness_funnel.png)

#### B. Multi-objective (ZDT1)

1. **NSGAParetoFrontPlot** — static scatter of the dominant (rank‑1) Pareto front.  
   _What-if cues:_ A thin front indicates strongly correlated objectives; broad clouds suggest good coverage but verify crowding to avoid gaps.  
   ![NSGAParetoFrontPlot (ZDT1)](./1-pareto_front.png)

2. **PopulationPlot** — plots the population envelope in objective space each generation.  
   _What-if cues:_ If the envelope flares without front improvement, survivors might be too permissive; a collapsing envelope hints at premature convergence.  
   ![PopulationPlot (ZDT1)](./ZDT1-1-population.png)

3. **OptPath** — traces decision-variable trajectories for the multi-objective run.  
   _What-if cues:_ Chaotic reruns around the final point usually mean conflicting constraints; smooth convergence shows stable selection pressure.  
   ![OptPath (ZDT1)](./ZDT1-1-opt_path.png)

4. **NSGACrowdingDistancePlot** — tracks crowding-distance statistics per generation.  
   _What-if cues:_ Collapsing percentiles imply diversity loss; erratic spikes may come from aggressive restarts or constraint filters.  
   ![CrowdingDistancePlot (ZDT1)](./1-crowding_distance.png)

5. **NSGARankHistoryPlot** — visualises the fraction of the population in each Pareto rank over time.  
   _What-if cues:_ Rank‑1 dominating early exposes premature convergence; oscillating ranks suggest survivor/offspring churn worth auditing.  
   ![RankHistoryPlot (ZDT1)](./1-rank_history.png)

6. **ParetoDiagnosticsPlot** — plots hypervolume progression and dominance counts across generations.  
   _What-if cues:_ Hypervolume interpretation depends on whether objectives are min/max (or mixed); set `<goals>` when using objective space or set `<space>fitness</space>` to base the plot on fitness columns.  
   ![ParetoDiagnosticsPlot (ZDT1)](./1-pareto_diagnostics.png)

7. **DominanceHeatMapPlot** — compares dominated and nondominated sample density in objective space.  
   _What-if cues:_ Dense dominated clusters beside sparse nondominated regions imply the search is circling good trade-offs without landing them.  
   ![DominanceHeatMapPlot (ZDT1)](./1-dominance_heatmap.png)

8. **TradeoffSlicePlot** — animates pairwise objective density contours with rank‑1 highlights.  
   _What-if cues:_ Bright contours devoid of orange points tell you elites aren’t exploring that slice; orange dots outside dense regions show elites outrunning the brood.  
   ![TradeoffSlicePlot (ZDT1 GIF)](./1-tradeoff_slices.gif)

9. **BubbleTradeoffPlot** — renders a two-objective scatter whose bubble size encodes a third metric.  
   _What-if cues:_ If large bubbles cluster in one corner, that metric trades off sharply with the objectives; scattered bubble sizes confirm balanced compromises.  
   ![BubbleTradeoffPlot (ZDT1)](./1-bubble_tradeoff.png)

10. **AttainmentSurfacePlot** — estimates empirical attainment probabilities across runs.  
    _What-if cues:_ Widely spaced contours imply inconsistent performance between runs; tightly nested levels show reliable attainment under the current settings.  
    ![AttainmentSurfacePlot (ZDT1)](./1-attainment_surface.png)

11. **ResponseSurfaceOverlayPlot** — overlays a smooth response surface with sampled points.  
    _What-if cues:_ Missing contours now fall back to scatter-only when data are sparse; if samples hug steep gradients, refine sampling around that ridge.  
    ![ResponseSurfaceOverlayPlot (ZDT1)](./1-response_surface_overlay.png)

12. **MultiRunUncertaintyPlot** — aggregates repeated runs and shades mean ± quantile bands.  
    _What-if cues:_ Wide bands mean run-to-run variability—tighten seeding or control randomness; shrinking bands with drifting mean reflects steady learning.  
    ![MultiRunUncertaintyPlot (ZDT1)](./1-multi_run_uncertainty.png)

13. **RadvizEmbeddingPlot** — projects high-dimensional variables to a radial Radviz layout.  
    _What-if cues:_ Rank‑1 samples sliding around the circle after changing generation filters show which anchors dominate late in the search.  
    ![RadvizEmbeddingPlot (ZDT1)](./1-radviz_embedding.png)

14. **ProsectionMatrixPlot** — slices variable pairs near the medians of remaining dimensions.  
    _What-if cues:_ If correlations vanish when tolerance tightens, relationships were driven by outliers; persistent bands confirm robust coupling.  
    ![ProsectionMatrixPlot (ZDT1)](./1-prosection_matrix.png)

15. **PreferenceSweepAnimationPlot** — animates how the best solution shifts while preference weights sweep between objectives.  
    _What-if cues:_ Rapid incumbent flips reveal sensitive trade-offs; long plateaus show stable choices even as stakeholders rebalance weights.  
    ![PreferenceSweepAnimationPlot (ZDT1 GIF)](./1-preference_sweep.gif)

16. **StarCoordinatesPlot** — renders multi-dimensional samples in a star-coordinates embedding.  
    _What-if cues:_ Nondominated rays collapsing toward one quadrant indicates dominance of specific variables; spreading rays across generations shows evolving trade-offs.  
    ![StarCoordinatesPlot (ZDT1)](./1-star_coordinates.png)

17. **SelfOrganizingMapPlot** — projects samples onto a self-organising map lattice to reveal neighbourhood occupancy.  
    _What-if cues:_ Empty regions highlight unexplored solution types; colour shifts between generations expose how constraint handling moves clusters.  
    ![SelfOrganizingMapPlot (ZDT1)](./1-som_map.png)

18. **ChordDiagramPlot** — draws chords between variables to highlight associations.  
    _What-if cues:_ Thick chords disappearing when you filter generations indicate transient correlations; persistent bands signal structural coupling worth modelling.  
    ![ChordDiagramPlot (ZDT1)](./1-chord_diagram.png)

19. **GlyphRadarPlot** — displays representative samples as radar glyphs for quick profile comparisons.  
    _What-if cues:_ If glyphs converge to identical shapes the population homogenised; distinct spikes pinpoint variables distinguishing elite designs under different scenarios.  
    ![GlyphRadarPlot (ZDT1)](./1-glyph_profiles.png)

## Remaining Visualization Wish-list

The following requests from the original survey are still open. Consider these when planning the next visualization sprint.

- [x] Self-organizing maps, star coordinates, chord diagrams, glyph-based plots, or Radviz-style embeddings — addressed via `SelfOrganizingMapPlot`, `StarCoordinatesPlot`, `ChordDiagramPlot`, `GlyphRadarPlot`, and `RadvizEmbeddingPlot`.
- [x] Prosection matrices or other interactive high-dimensional slicing — addressed via `ProsectionMatrixPlot`.
- [x] Preference exploration animations (weight sweeping, constraint relaxation) — addressed via `PreferenceSweepAnimationPlot`.

## Maintenance Notes

- `DiversityRadarPlot` is implemented but lacks a regression harness; wire it into a multi-objective test (for example the ZDT suite) before relying on it in production workflows.
- Keep regression artifacts in `tests/framework/Optimizers/...` updated with `./run_tests --re=<pattern> --update` whenever plot output formats change.
- ImageIO v3 deprecation warnings are already mitigated in the animation plotters by importing `imageio.v2`; maintain that convention when adding new animated plots.
