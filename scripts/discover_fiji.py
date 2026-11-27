"""Discover which Fiji commands, classes and MorphoLibJ backends work headless (Requirement 6).

Run once after creating the environment, and again after any Fiji upgrade:

    python scripts/discover_fiji.py [--out docs/fiji_discovery.json]

The findings are copied by hand into the `mozzavid/fiji.py` constants. Nothing reads the JSON
at runtime (Requirement 6.7). Every probe catches and records its own exception, so one failing
backend cannot hide what the other probes found.

Probe images are built here voxel by voxel with `ij.ImageStack` rather than through the
`fiji.py` conversion helpers, so discovery only depends on `get_ij()` and `runtime_versions()`.
`scyjava` is reached through the `fiji` module because only `fiji.py` may import it (Req 27.2).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/discover_fiji.py` without an install
    sys.path.insert(0, str(REPO_ROOT))

from mozzavid import fiji  # noqa: E402  (import after the sys.path fix above)

DEFAULT_OUT = Path("docs") / "fiji_discovery.json"

COMMAND_KEYWORDS = ("morpholibj", "ijpb", "connected component", "label", "region", "3d", "mcib")
SHAPE_COLUMN_KEYWORDS = ("feret", "ellipsoid", "elli.", "elong", "azim", "elev", "roll", "radius")

PROBE_SIZE = 16                                   # 16³ probe volume (Requirement 6.4)
SPHERE_RADIUS = 2.5                               # ~65 voxels each; a 2-voxel gap separates them
SPHERE_CENTRES_ZYX = ((8, 8, 4), (8, 8, 11))
CUBE_RANGES = (range(2, 5), range(5, 8))          # 3³ cubes that share only a corner diagonal (H6)
PROBE_MIN_SIZE = 10                               # size-opening threshold below each sphere's size
LABEL_BIT_DEPTH = 16
EXPECTED_SPHERE_COMPONENTS = 2

# Candidate names to test, not facts (design "Component: fiji.py"). Discovery decides.
CANDIDATE_COMMANDS = {
    "connected_components": "Connected Components Labeling",
    "size_opening": "Label Size Opening",
    "analyze_regions_3d": "Analyze Regions 3D",
}
ANALYZE_REGIONS_3D_ARGS = (
    "voxel_count volume surface_area mean_breadth sphericity euler_number bounding_box "
    "centroid equivalent_ellipsoid ellipsoid_elongations max._inscribed "
    "surface_area_method=[Crofton (13 dirs.)] euler_connectivity=26"
)
MCIB3D_CLASSES = (
    "mcib3d.image3d.ImageHandler",
    "mcib3d.image3d.ImageLabeller",
    "mcib3d.geom2.Objects3DIntPopulation",
)
MCIB3D_MEASURE_CLASSES = tuple(
    f"mcib3d.geom2.measurements.{name}"
    for name in ("MeasureVolume", "MeasureSurface", "MeasureCompactness",
                 "MeasureEllipsoid", "MeasureFeret", "MeasureCentroid")
)


# --------------------------------------------------------------------------- probe plumbing

def _jimport(class_name: str):
    """Import a Java class with scyjava.jimport, reached through the gateway module."""
    return fiji.scyjava.jimport(class_name)


def _failure(error: BaseException) -> dict[str, Any]:
    """Describe a failed probe step, keeping a short traceback for diagnosis."""
    return {
        "ok": False,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(limit=6),
    }


def _attempt(probe: Callable[..., dict], *args) -> dict[str, Any]:
    """Run one probe, returning {"ok": True, **result} or the recorded failure."""
    try:
        return {"ok": True, **probe(*args)}
    except Exception as error:  # JPype Java exceptions derive from Exception
        return _failure(error)


def _step(record: dict, name: str, operation: Callable, argument, describe: Callable):
    """Run one chained backend step and store its outcome in record[name].

    Returns the step result, or None when the step failed or was skipped because an earlier
    step produced no input, so later steps report "skipped" instead of a misleading error.
    """
    if argument is None:
        record[name] = {"ok": False, "error": "skipped: an earlier step failed"}
        return None
    try:
        result = operation(argument)
    except Exception as error:
        record[name] = _failure(error)
        return None
    record[name] = {"ok": True, **describe(result)}
    return result


# --------------------------------------------------------------------------- probe images

def _sphere_voxel(z: int, y: int, x: int) -> bool:
    """True inside either of the two separated probe spheres."""
    return any(
        (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= SPHERE_RADIUS ** 2
        for cz, cy, cx in SPHERE_CENTRES_ZYX
    )


def _cube_voxel(z: int, y: int, x: int) -> bool:
    """True inside either cube; the cubes touch only across the (4,4,4)-(5,5,5) diagonal."""
    return any(z in span and y in span and x in span for span in CUBE_RANGES)


def _build_binary_imageplus(title: str, is_foreground: Callable[[int, int, int], bool]):
    """Build a 16³ 8-bit 0/255 ImagePlus with one stack slice per z.

    Per-voxel Java calls are slow but harmless for 4096 voxels, and they avoid any
    dependency on the numpy conversion helpers.
    """
    image_stack_class = _jimport("ij.ImageStack")
    byte_processor_class = _jimport("ij.process.ByteProcessor")
    stack = image_stack_class(PROBE_SIZE, PROBE_SIZE)
    for z in range(PROBE_SIZE):
        processor = byte_processor_class(PROBE_SIZE, PROBE_SIZE)
        for y in range(PROBE_SIZE):
            for x in range(PROBE_SIZE):
                if is_foreground(z, y, x):
                    processor.set(x, y, 255)
        stack.addSlice(f"z{z}", processor)
    return _jimport("ij.ImagePlus")(title, stack)


def _label_values(imp) -> list[int]:
    """Return the sorted distinct non-zero voxel values of an ImagePlus stack."""
    stack = imp.getStack()
    values = {
        int(stack.getVoxel(x, y, z))
        for z in range(stack.getSize())
        for y in range(stack.getHeight())
        for x in range(stack.getWidth())
    }
    values.discard(0)
    return sorted(values)


def _describe_labels(imp) -> dict[str, Any]:
    """Summarise a label image: class, bit depth, slices and component count."""
    labels = _label_values(imp)
    return {
        "image_class": str(imp.getClass().getName()),
        "bit_depth": int(imp.getBitDepth()),
        "n_slices": int(imp.getNSlices()),
        "n_components": len(labels),
        "expected_components": EXPECTED_SPHERE_COMPONENTS,
    }


def _table_columns(table) -> tuple[list[str], int]:
    """Return (column names, row count) for an ImageJ ResultsTable or a SciJava Table."""
    if hasattr(table, "getHeadings"):
        return [str(heading) for heading in table.getHeadings()], int(table.size())
    columns = [str(table.getColumnHeader(index)) for index in range(table.getColumnCount())]
    return columns, int(table.getRowCount())


def _describe_table(table) -> dict[str, Any]:
    """Record the table class, its columns and any Feret / ellipsoid shape columns (H7)."""
    columns, n_rows = _table_columns(table)
    shape_columns = [c for c in columns if any(k in c.lower() for k in SHAPE_COLUMN_KEYWORDS)]
    return {
        "table_class": str(table.getClass().getName()),
        "columns": columns,
        "n_rows": n_rows,
        "shape_columns": shape_columns,
        "has_feret_column": any("feret" in c.lower() for c in columns),
        "has_ellipsoid_columns": any("elli" in c.lower() for c in columns),
    }


# --------------------------------------------------------------------------- environment probes

def _probe_headless(ij) -> dict[str, Any]:
    """Record whether AWT and the ImageJ UI service both report headless mode."""
    graphics_environment = _jimport("java.awt.GraphicsEnvironment")
    return {
        "awt_headless": bool(graphics_environment.isHeadless()),
        "ui_headless": bool(ij.ui().isHeadless()),
    }


def _matches_keywords(text: str) -> bool:
    """True when the text contains a MorphoLibJ or 3D-analysis keyword."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in COMMAND_KEYWORDS)


def _scijava_commands(ij) -> list[dict[str, str]]:
    """List SciJava CommandInfos whose title or class matches the keywords."""
    found = []
    for info in ij.command().getCommands():
        title, class_name = str(info.getTitle()), str(info.getClassName())
        if _matches_keywords(f"{title} {class_name}"):
            found.append({"title": title, "class": class_name, "menu_path": str(info.getMenuPath())})
    return sorted(found, key=lambda entry: entry["title"])


def _ij1_commands() -> dict[str, str]:
    """Map ImageJ 1.x menu command labels to plugin classes, filtered by keyword.

    MorphoLibJ registers ImageJ 1.x plugins, so its commands appear in ij.Menus.
    """
    table = _jimport("ij.Menus").getCommands()
    if table is None:
        return {}
    pairs = {str(key): str(table.get(key)) for key in table.keySet()}
    return {label: cls for label, cls in sorted(pairs.items()) if _matches_keywords(f"{label} {cls}")}


def _probe_commands(ij) -> dict[str, Any]:
    """List matching registered commands and check each candidate command string."""
    ij1 = _ij1_commands()
    scijava = _scijava_commands(ij)
    scijava_titles = {entry["title"] for entry in scijava}
    registered = {
        role: {"command": name, "ij1": name in ij1, "scijava": name in scijava_titles}
        for role, name in CANDIDATE_COMMANDS.items()
    }
    return {"ij1_commands": ij1, "scijava_commands": scijava, "candidates": registered}


def _probe_class(class_name: str) -> dict[str, Any]:
    """jimport a class and list its public non-Object method signatures and fields."""
    java_class = _jimport(class_name).class_
    methods = sorted(
        str(method.toGenericString())
        for method in java_class.getMethods()
        if str(method.getDeclaringClass().getName()) != "java.lang.Object"
    )
    fields = sorted(str(field.getName()) for field in java_class.getFields())
    return {"class": class_name, "methods": methods, "fields": fields}


def _morpholibj_class_names() -> tuple[str, ...]:
    """The CLASS_* constants from fiji.py plus the MorphometricFeatures3D feature enum."""
    return (
        fiji.CLASS_BINARY_IMAGES,
        fiji.CLASS_LABEL_IMAGES,
        fiji.CLASS_ANALYZE_REGIONS_3D,
        f"{fiji.CLASS_ANALYZE_REGIONS_3D}$Feature",
        fiji.CLASS_MAX_FERET_3D,
    )


# --------------------------------------------------------------------------- "plugin" backend

def _current_image_other_than(input_imp):
    """Return the current ImageJ image, failing if the plugin did not produce a new one."""
    current = _jimport("ij.WindowManager").getCurrentImage()
    if current is None or str(current.getTitle()) == str(input_imp.getTitle()):
        raise RuntimeError("plugin produced no new current image (WindowManager) in headless mode")
    return current


def _plugin_label(ij, imp, connectivity: int):
    """Connected-component labelling via ij.py.run_plugin with macro-style args."""
    args = f"connectivity={connectivity} type=[{LABEL_BIT_DEPTH} bits]"
    ij.py.run_plugin(CANDIDATE_COMMANDS["connected_components"], args, imp=imp)
    return _current_image_other_than(imp)


def _plugin_size_opening(ij, labels):
    """Label size opening via ij.py.run_plugin."""
    ij.py.run_plugin(CANDIDATE_COMMANDS["size_opening"], f"min={PROBE_MIN_SIZE}", imp=labels)
    return _current_image_other_than(labels)


def _plugin_analyze_regions(ij, labels):
    """Analyze Regions 3D via ij.py.run_plugin; MorphoLibJ names its table '<title>-morpho'."""
    ij.py.run_plugin(CANDIDATE_COMMANDS["analyze_regions_3d"], ANALYZE_REGIONS_3D_ARGS, imp=labels)
    results_table_class = _jimport("ij.measure.ResultsTable")
    table = results_table_class.getResultsTable(f"{labels.getShortTitle()}-morpho")
    if table is None:
        raise RuntimeError("no '<title>-morpho' ResultsTable is reachable in headless mode")
    return table


def _plugin_operations(ij) -> dict[str, Callable]:
    """Bind the plugin-backend steps to the ImageJ gateway."""
    return {
        "label": lambda imp, connectivity: _plugin_label(ij, imp, connectivity),
        "size_opening": lambda labels: _plugin_size_opening(ij, labels),
        "analyze": lambda labels: _plugin_analyze_regions(ij, labels),
    }


# --------------------------------------------------------------------------- "direct" backend

def _direct_label(imp, connectivity: int):
    """BinaryImages.componentsLabeling(ImagePlus, connectivity, bitDepth)."""
    binary_images = _jimport(fiji.CLASS_BINARY_IMAGES)
    return binary_images.componentsLabeling(imp, connectivity, LABEL_BIT_DEPTH)


def _direct_size_opening(labels):
    """LabelImages.sizeOpening(ImagePlus, minVoxelCount)."""
    return _jimport(fiji.CLASS_LABEL_IMAGES).sizeOpening(labels, PROBE_MIN_SIZE)


def _direct_analyze_regions(labels):
    """MorphometricFeatures3D(all Feature values).computeTable(ImagePlus) -> ResultsTable.

    MorphoLibJ 1.6.3's AnalyzeRegions3D plugin only exposes run(String); this is the class
    it delegates to. All enum values are requested so every available column is listed.
    """
    analyzer_class = _jimport(fiji.CLASS_ANALYZE_REGIONS_3D)
    features = _jimport(f"{fiji.CLASS_ANALYZE_REGIONS_3D}$Feature").values()
    return analyzer_class(features).computeTable(labels)


def _probe_max_feret(labels) -> dict[str, Any]:
    """H7: run MaxFeretDiameter3D on the probe labels and describe its table."""
    return _describe_table(_jimport(fiji.CLASS_MAX_FERET_3D)().computeTable(labels))


def _direct_operations() -> dict[str, Callable]:
    """The direct-backend steps (scyjava calls into inra.ijpb.* classes)."""
    return {"label": _direct_label, "size_opening": _direct_size_opening, "analyze": _direct_analyze_regions}


# --------------------------------------------------------------------------- shared backend probe

def _cube_probe(label_operation: Callable) -> dict[str, Any]:
    """H6: count components of the corner-touching cubes at CONNECTIVITY_3D and at 6."""
    counts = {}
    for connectivity in sorted({fiji.CONNECTIVITY_3D, 6}):
        cubes = _build_binary_imageplus(f"probe-cubes-c{connectivity}", _cube_voxel)
        counts[str(connectivity)] = len(_label_values(label_operation(cubes, connectivity)))
    return {
        "connectivity_3d": fiji.CONNECTIVITY_3D,
        "components_by_connectivity": counts,
        "merged_at_connectivity_3d": counts[str(fiji.CONNECTIVITY_3D)] == 1,
    }


def _probe_backend(operations: dict[str, Callable]) -> dict[str, Any]:
    """Run label -> size opening -> region analysis on the two-sphere probe, then the cube probe.

    `headless_ok` is True only if all three steps succeed (Requirement 6.4, H5).
    """
    spheres = _build_binary_imageplus("probe-spheres", _sphere_voxel)
    record: dict[str, Any] = {}
    label_at_default = lambda imp: operations["label"](imp, fiji.CONNECTIVITY_3D)  # noqa: E731
    labels = _step(record, "label", label_at_default, spheres, _describe_labels)
    opened = _step(record, "size_opening", operations["size_opening"], labels, _describe_labels)
    _step(record, "analyze_regions_3d", operations["analyze"], opened, _describe_table)
    record["headless_ok"] = all(record[s]["ok"] for s in ("label", "size_opening", "analyze_regions_3d"))
    if labels is None:
        record["cube_probe"] = {"ok": False, "error": "skipped: labelling failed"}
    else:
        record["cube_probe"] = _attempt(_cube_probe, operations["label"])
    return record


# --------------------------------------------------------------------------- mcib3d fallback

def _mcib3d_label(imp, connectivity: int):
    """Label with mcib3d ImageLabeller. It has no connectivity argument, so it is ignored."""
    handler = _jimport("mcib3d.image3d.ImageHandler").wrap(imp)
    return _jimport("mcib3d.image3d.ImageLabeller")().getLabels(handler).getImagePlus()


def _mcib3d_size_opening(labels):
    """Re-label with a minimum object size, mcib3d's equivalent of a size opening."""
    labeller = _jimport("mcib3d.image3d.ImageLabeller")()
    labeller.setMinSize(PROBE_MIN_SIZE)
    handler = _jimport("mcib3d.image3d.ImageHandler").wrap(labels)
    return labeller.getLabels(handler).getImagePlus()


def _mcib3d_measure_names(first_object) -> list[str]:
    """Collect measurement names from each importable mcib3d Measure* class."""
    names: list[str] = []
    for class_name in MCIB3D_MEASURE_CLASSES:
        try:
            measure = _jimport(class_name)(first_object)
            names.extend(f"{class_name.rsplit('.', 1)[-1]}:{n}" for n in measure.getNamesMeasurement())
        except Exception:  # an absent Measure class only removes its columns
            continue
    return names


class _Mcib3dTable:
    """Duck-typed ResultsTable stand-in so mcib3d names flow through `_describe_table`.

    mcib3d measures per object rather than returning a table, so `getClass()` returns this
    object and `getName()` reports where the column names came from.
    """

    def __init__(self, columns: list[str], n_objects: int):
        """Store the measurement names and the population size."""
        self._columns, self._n_objects = columns, n_objects

    def getHeadings(self):  # noqa: N802  (mirrors the ResultsTable method name)
        """Return the measurement names."""
        return self._columns

    def size(self):
        """Return the number of measured objects."""
        return self._n_objects

    def getClass(self):  # noqa: N802
        """Return self, standing in for java.lang.Class."""
        return self

    def getName(self):  # noqa: N802
        """Name the source of the columns in place of a Java table class name."""
        return "mcib3d.geom2.measurements.Measure* (per-object, no ResultsTable)"


def _mcib3d_analyze(labels):
    """Build an Objects3DIntPopulation and collect mcib3d per-object measurement names."""
    handler = _jimport("mcib3d.image3d.ImageHandler").wrap(labels)
    population = _jimport("mcib3d.geom2.Objects3DIntPopulation")(handler)
    objects = list(population.getObjects3DInt())
    if not objects:
        raise RuntimeError("mcib3d population is empty")
    return _Mcib3dTable(_mcib3d_measure_names(objects[0]), len(objects))


def _mcib3d_operations() -> dict[str, Callable]:
    """The 3D ImageJ Suite steps, run through the same probe as MorphoLibJ (Req 6.8)."""
    return {"label": _mcib3d_label, "size_opening": _mcib3d_size_opening, "analyze": _mcib3d_analyze}


def _maybe_probe_mcib3d(backends: dict[str, dict]) -> dict[str, Any]:
    """Probe the 3D ImageJ Suite only when neither MorphoLibJ backend works headless."""
    if any(result.get("headless_ok") for result in backends.values()):
        return {"probed": False, "reason": "a MorphoLibJ backend runs headless"}
    return {
        "probed": True,
        "classes": {name: _attempt(_probe_class, name) for name in MCIB3D_CLASSES},
        "direct": _attempt(_probe_backend, _mcib3d_operations()),
    }


# --------------------------------------------------------------------------- summary and report

def _summarise(findings: dict[str, Any]) -> dict[str, Any]:
    """Condense the findings into what must be pinned in fiji.py and the H5/H6 outcomes.

    "direct" is listed first when both work because it avoids WindowManager global state.
    """
    backends = findings.get("backends", {})
    working = [name for name in ("direct", "plugin") if backends.get(name, {}).get("headless_ok")]
    mcib3d_ok = bool(findings.get("mcib3d", {}).get("direct", {}).get("headless_ok"))
    cube = next((backends[name]["cube_probe"] for name in working), {})
    classes = findings.get("classes", {})
    return {
        "working_morpholibj_backends": working,
        "suggested_backend": working[0] if working else None,
        "morpholibj_importable": bool(classes.get(fiji.CLASS_BINARY_IMAGES, {}).get("ok")),
        "mcib3d_headless_ok": mcib3d_ok,
        "h5_supported": bool(working),
        "h6_cubes_merged_at_connectivity_3d": cube.get("merged_at_connectivity_3d"),
        "stop_and_report": not working and not mcib3d_ok,
    }


def _start_and_report_versions() -> dict[str, str]:
    """Start Fiji without the constant check (discovery is how stale constants get fixed)."""
    fiji.get_ij(verify=False)
    return fiji.runtime_versions()


def discover() -> dict[str, Any]:
    """Run every probe and return the findings; the JVM is started once via fiji.get_ij()."""
    findings: dict[str, Any] = {
        "fiji_coordinate": fiji.FIJI_COORDINATE,
        "extra_endpoints": list(fiji.EXTRA_ENDPOINTS),
        "candidate_commands": CANDIDATE_COMMANDS,
        "analyze_regions_3d_args": ANALYZE_REGIONS_3D_ARGS,
    }
    findings["versions"] = _attempt(_start_and_report_versions)
    if findings["versions"]["ok"]:  # without a running JVM nothing else can be probed
        _run_jvm_probes(findings)
    findings["summary"] = _summarise(findings)
    return findings


def _run_jvm_probes(findings: dict[str, Any]) -> None:
    """Run every probe that needs a started JVM and store the results in `findings`."""
    ij = fiji.get_ij()
    findings["headless"] = _attempt(_probe_headless, ij)
    findings["commands"] = _attempt(_probe_commands, ij)
    findings["classes"] = {name: _attempt(_probe_class, name) for name in _morpholibj_class_names()}
    findings["backends"] = {
        "plugin": _attempt(_probe_backend, _plugin_operations(ij)),
        "direct": _attempt(_probe_backend, _direct_operations()),
    }
    findings["mcib3d"] = _maybe_probe_mcib3d(findings["backends"])
    findings["max_feret_3d"] = _attempt(
        lambda: _probe_max_feret(_direct_label(
            _build_binary_imageplus("probe-feret", _sphere_voxel), fiji.CONNECTIVITY_3D)))


def _status(result: dict[str, Any], key: str = "ok") -> str:
    """Render a probe outcome as OK or FAIL plus the first line of its error."""
    if result.get(key):
        return "OK"
    return f"FAIL ({str(result.get('error', 'not ok')).splitlines()[0]})"


def _report_backend(name: str, result: dict[str, Any]) -> None:
    """Print one backend's step outcomes, table class and columns."""
    print(f"  backend {name}: headless_ok={result.get('headless_ok')}  {_status(result)}")
    for step in ("label", "size_opening", "analyze_regions_3d", "cube_probe"):
        if step in result:
            print(f"    {step:<20} {_status(result[step])}")
    table = result.get("analyze_regions_3d", {})
    if table.get("ok"):
        print(f"    table class: {table['table_class']}")
        print(f"    columns: {', '.join(table['columns'])}")
        print(f"    shape columns (Feret/ellipsoid): {table['shape_columns']}")
    if result.get("cube_probe", {}).get("ok"):
        print(f"    H6 components by connectivity: {result['cube_probe']['components_by_connectivity']}")


def print_report(findings: dict[str, Any], out_path: Path) -> None:
    """Print a console summary of the findings (Requirement 6.6)."""
    versions = findings["versions"]
    print(f"Fiji coordinate: {findings['fiji_coordinate']}")
    print(f"Versions: {_status(versions)} " + ", ".join(
        f"{k}={versions[k]}" for k in ("fiji", "java", "pyimagej", "scyjava") if k in versions))
    commands = findings.get("commands", {})
    for role, entry in commands.get("candidates", {}).items():
        print(f"  command {role:<22} '{entry['command']}' ij1={entry['ij1']} scijava={entry['scijava']}")
    print(f"  matching IJ1 commands: {len(commands.get('ij1_commands', {}))}, "
          f"SciJava commands: {len(commands.get('scijava_commands', []))}")
    for name, result in findings.get("classes", {}).items():
        print(f"  class {name}: {_status(result)}")
    for name, result in findings.get("backends", {}).items():
        _report_backend(name, result)
    if findings.get("mcib3d", {}).get("probed"):
        _report_backend("mcib3d", findings["mcib3d"]["direct"])
    print(f"Summary: {json.dumps(findings['summary'])}")
    print(f"Findings written to {out_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse --out (default docs/fiji_discovery.json, Requirement 21.7)."""
    parser = argparse.ArgumentParser(description="Discover headless Fiji/MorphoLibJ capabilities.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"JSON output path (default: {DEFAULT_OUT.as_posix()})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run discovery, write the JSON, print the report; exit 1 when stop-and-report applies."""
    args = _parse_args(argv)
    findings = discover()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print_report(findings, args.out)
    if findings["summary"]["stop_and_report"]:
        print("STOP AND REPORT: no backend runs labelling, size opening and region analysis "
              "headless (H5). Do not substitute scikit-image.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
