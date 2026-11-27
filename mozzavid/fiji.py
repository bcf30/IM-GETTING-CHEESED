"""The only gateway to the JVM: Fiji version pins, plugin/command strings and the ImageJ singleton.

This is the only module allowed to import `imagej` and `scyjava` (Requirement 27.2), and it
holds every Fiji version pin and every plugin, command and class string (Requirement 27.4).
The pins and command/class constants were copied by hand from `scripts/discover_fiji.py` output
(recorded in docs/reproducibility.md); they are never loaded from JSON at runtime, so behaviour
stays deterministic and reviewable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version

import imagej
import numpy as np
import pandas as pd
import scyjava
import xarray as xr  # only imported here (Requirement 24.4); PyImageJ maps these dim names to axes

log = logging.getLogger(__name__)

# 2.17.0 is the newest release that runs on the pinned openjdk 11: 2.18.0 ships Java 21 class
# files (discovery: UnsupportedClassVersionError, class file 65). See docs/reproducibility.md.
FIJI_COORDINATE: str = "sc.fiji:fiji:2.17.0"
# Discovery found MorphoLibJ absent from sc.fiji:fiji (Req 6.14). Its Maven group is fr.inra.ijpb
# (sc.fiji:MorphoLibJ_ does not exist); 1.6.3 is the version managed by Fiji 2.17.0's pom-scijava 42.
EXTRA_ENDPOINTS: list[str] = ["fr.inra.ijpb:MorphoLibJ_:1.6.3"]

JVM_MAX_HEAP: str = "-Xmx4g"                  # 192³ float32 = 28 MB; several copies + labels fit easily
CONNECTIVITY_3D: int = 26                     # matches ImageJ 3D Objects Counter convention; see H6

# Backend selection, set from discovery. "direct" = scyjava calls into inra.ijpb.* classes;
# "plugin" = ij.py.run_plugin with macro-style args. Only "direct" passed all three steps
# headless: the plugin's Analyze Regions 3D table is not reachable without a GUI.
MORPHOLIBJ_BACKEND: str = "direct"
# Registered IJ1 menu commands (discovery: ij1=True). The direct backend calls the classes
# below; these names are checked so a Fiji upgrade that drops MorphoLibJ fails fast.
CMD_CONNECTED_COMPONENTS: str = "Connected Components Labeling"
CMD_SIZE_OPENING: str = "Label Size Opening"
CMD_ANALYZE_REGIONS_3D: str = "Analyze Regions 3D"
CLASS_BINARY_IMAGES: str = "inra.ijpb.binary.BinaryImages"
CLASS_LABEL_IMAGES: str = "inra.ijpb.label.LabelImages"
# MorphoLibJ 1.6.3's AnalyzeRegions3D plugin exposes only run(String); the class it delegates to is:
CLASS_ANALYZE_REGIONS_3D: str = "inra.ijpb.measure.region3d.MorphometricFeatures3D"
CLASS_MAX_FERET_3D: str = "inra.ijpb.measure.region3d.MaxFeretDiameter3D"  # importable but unused (H7)

# Float labels hold up to 2^24 exact integers (> 192³ voxels), so labelling a noisy mask cannot
# overflow, which 16-bit labels (65535) could.
LABEL_BIT_DEPTH: int = 32
# MorphometricFeatures3D$Feature values requested; the inscribed ball is skipped (unused, costly).
MORPHOMETRIC_FEATURES_3D: tuple[str, ...] = (
    "VOXEL_COUNT", "VOLUME", "SURFACE_AREA", "SPHERICITY", "BOUNDING_BOX", "CENTROID",
    "EQUIVALENT_ELLIPSOID",
)
MAX_FERET_COLUMN: str = "MaxFeret.Diameter"   # MaxFeretDiameter3D's column; no longer used (H7)
_FERET_DIAMETER_COLUMN: str = "Diameter"      # column name in MaxFeretDiameter3D's own table (H7)

# Fiji column name -> internal raw name (pixel units), from the discovery probe table.
COLUMN_MAP: dict[str, str] = {
    "Label": "label",
    "VoxelCount": "voxel_count",
    "Volume": "volume_px3",
    "SurfaceArea": "surface_area_px2",
    "Sphericity": "sphericity_morpholibj",   # 36πV²/S³ = Wadell³ (docs/reproducibility.md)
    "Box.X.Min": "bbox_x_min_px", "Box.X.Max": "bbox_x_max_px",
    "Box.Y.Min": "bbox_y_min_px", "Box.Y.Max": "bbox_y_max_px",
    "Box.Z.Min": "bbox_z_min_px", "Box.Z.Max": "bbox_z_max_px",
    "Centroid.X": "centroid_x_px", "Centroid.Y": "centroid_y_px", "Centroid.Z": "centroid_z_px",
    "Elli.Radius1": "ellipsoid_radius1_px", "Elli.Radius2": "ellipsoid_radius2_px",
    "Elli.Radius3": "ellipsoid_radius3_px",
}

_JAVA_VERSION_COMMAND: tuple[str, ...] = ("java", "-version")
_MAVEN_VERSION_COMMAND: tuple[str, ...] = ("mvn", "-v")
_DIAGNOSTIC_TIMEOUT_S: int = 30

# PyImageJ dim names: "pln" -> Z, "row" -> Y, "col" -> X, so numpy (Z,Y,X) becomes Java (X,Y,Z).
ZYX_DIMS: tuple[str, str, str] = ("pln", "row", "col")
MASK_FOREGROUND: int = 255                    # 8-bit binary convention expected by MorphoLibJ
_COLUMN_NOT_FOUND: int = -1                   # ij.measure.ResultsTable.COLUMN_NOT_FOUND

# Process-level singleton (design "Process-level state"): the JVM can start only once per
# process and imagej.init costs 15-20 s. It holds no per-volume data.
_IJ = None


def _command_output(command: tuple[str, ...]) -> str:
    """Return the combined stdout/stderr of a diagnostic command, or why it could not run.

    The executable is resolved with shutil.which because on Windows Maven is `mvn.cmd`,
    which subprocess does not find from the bare name. `java -version` writes to stderr,
    so both streams are captured.
    """
    executable = shutil.which(command[0])
    if executable is None:
        return f"`{' '.join(command)}`: executable not found on PATH"
    try:
        completed = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True,
            timeout=_DIAGNOSTIC_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"`{' '.join(command)}`: failed to run ({error})"
    return (completed.stdout + completed.stderr).strip()


def _init_failure_message(error: BaseException) -> str:
    """Build the RuntimeError text for a failed ImageJ start (Requirement 5.4).

    Init failures are almost always a missing/wrong Java or Maven, or no network on the
    first Maven download, so the toolchain versions are included to make the cause visible.
    """
    return (
        f"Failed to initialise Fiji from Maven coordinate {FIJI_COORDINATE} "
        f"(extra endpoints: {EXTRA_ENDPOINTS}): {error}\n"
        f"--- java -version ---\n{_command_output(_JAVA_VERSION_COMMAND)}\n"
        f"--- mvn -v ---\n{_command_output(_MAVEN_VERSION_COMMAND)}"
    )


def _ensure_conda_java_home() -> None:
    """Point JAVA_HOME and PATH at the conda env's bundled JDK when it is present and complete.

    scyjava's fetch="auto" otherwise downloads a cjdk JRE that on this machine is incomplete
    (no jar.exe), so imagej.init fails intermittently with 'NoneType is not callable', depending
    on whether jgo's PATH search happens to find the conda Java first. The conda openjdk pinned
    in environment.yml is a complete JDK under the interpreter prefix. Setting JAVA_HOME and
    prepending its bin to PATH makes jgo's "system Java" detection deterministic, so the JVM is
    always the conda JDK regardless of how Python was launched. An existing JAVA_HOME the user set
    is left untouched.
    """
    import os
    import sys
    candidate = os.path.join(sys.prefix, "Library", "lib", "jvm")
    bin_dir = os.path.join(candidate, "bin")
    if not (os.path.isfile(os.path.join(bin_dir, "java.exe"))
            and os.path.isfile(os.path.join(bin_dir, "jar.exe"))):
        return
    os.environ.setdefault("JAVA_HOME", candidate)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    log.info("using the conda env JDK: %s", candidate)


def _start_imagej():
    """Configure the JVM heap and start headless Fiji with the legacy layer.

    The heap option must be added before the JVM starts. A single coordinate string is
    passed when there are no extra endpoints, matching the plain `imagej.init` usage.
    fetch="auto" makes scyjava use the conda-pinned openjdk (JAVA_HOME) instead of its default
    "always" download of a separate JRE; "never" trips a pyimagej 1.8.0 version-guess bug.
    """
    _ensure_conda_java_home()
    scyjava.config.set_java_constraints(fetch="auto")
    scyjava.config.add_option(JVM_MAX_HEAP)
    endpoints = [FIJI_COORDINATE, *EXTRA_ENDPOINTS]
    log.info("Starting Fiji %s headless (first call takes 15-20 s)", endpoints)
    init_target = endpoints if EXTRA_ENDPOINTS else FIJI_COORDINATE
    try:
        return imagej.init(init_target, mode="headless", add_legacy=True)
    except Exception as error:  # any init failure is reported with toolchain diagnostics
        raise RuntimeError(_init_failure_message(error)) from error


def get_ij(verify: bool = True):
    """Return the single headless ImageJ gateway, initialising the JVM on first call.

    Later calls return the same object, so the JVM start-up cost is paid once per process.
    On the first call the command/class constants are checked (Req 6.12) unless
    `verify=False`, which only `scripts/discover_fiji.py` uses, since it exists to find them.
    A failed start or check leaves the singleton unset and raises RuntimeError.
    """
    global _IJ
    if _IJ is None:
        ij = _start_imagej()
        if verify:
            verify_constants()
        _IJ = ij
    return _IJ


def _missing_commands() -> list[str]:
    """Return the CMD_* constants that are not registered ImageJ 1.x menu commands."""
    registered = scyjava.jimport("ij.Menus").getCommands()
    commands = (CMD_CONNECTED_COMPONENTS, CMD_SIZE_OPENING, CMD_ANALYZE_REGIONS_3D)
    return [name for name in commands if registered is None or not registered.containsKey(name)]


def _missing_classes() -> list[str]:
    """Return the CLASS_* constants that scyjava cannot import."""
    missing = []
    for name in (CLASS_BINARY_IMAGES, CLASS_LABEL_IMAGES, CLASS_ANALYZE_REGIONS_3D,
                 CLASS_MAX_FERET_3D):
        try:
            scyjava.jimport(name)
        except Exception:  # JPype raises TypeError/ImportError variants for unknown classes
            missing.append(name)
    return missing


def verify_constants() -> None:
    """Raise RuntimeError naming every unregistered command or non-importable class (Req 6.12).

    Needs a started JVM. A Fiji or MorphoLibJ upgrade that renames something fails here,
    before any volume is processed.
    """
    commands, classes = _missing_commands(), _missing_classes()
    if commands or classes:
        raise RuntimeError(
            f"Fiji constants do not resolve in {FIJI_COORDINATE} + {EXTRA_ENDPOINTS}: "
            f"unregistered commands {commands}, non-importable classes {classes}. "
            "Run `python scripts/discover_fiji.py` and update the constants in mozzavid/fiji.py."
        )


def _package_version(distribution_name: str) -> str:
    """Return an installed Python distribution's version, or "unknown" if not installed."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def runtime_versions() -> dict[str, str]:
    """Return the Fiji, Java, PyImageJ and scyjava versions as strings (Requirement 5.8).

    Starts the JVM if needed, because the Fiji and Java versions are only known at runtime;
    they are recorded for reproducibility.
    """
    ij = get_ij()
    java_system = scyjava.jimport("java.lang.System")
    return {
        "fiji": str(ij.getVersion()),
        "java": str(java_system.getProperty("java.version")),
        "pyimagej": _package_version("pyimagej"),
        "scyjava": _package_version("scyjava"),
    }


# --- numpy <-> Java conversion (Requirement 7) -------------------------------------------
# Every helper builds its Java objects per call and returns them; nothing is cached here.


def collect_garbage() -> None:
    """Prompt the JVM to free image memory between volumes (Requirement 13.10).

    PyImageJ/ImgLib2 conversions leave short-lived Java objects (a full 192³ float image is
    ~28 MB) that the JVM reclaims lazily. Over a long sequential run the 4 GB heap can fill
    before a GC cycle runs, raising OutOfMemoryError. Calling System.gc() after each volume
    keeps peak heap flat; it is cheap relative to the ~2 s per-volume processing time.
    """
    scyjava.jimport("java.lang.System").gc()


def _require_3d(arr: np.ndarray) -> None:
    """Reject non-3D input at the boundary, because axis semantics are only defined for (Z,Y,X)."""
    if np.ndim(arr) != 3:
        raise ValueError(f"expected a 3D (Z,Y,X) array, got shape {np.shape(arr)}")


def to_java_img(arr: np.ndarray):
    """Convert a numpy (Z,Y,X) array to a Java image with dimensions (X,Y,Z) (Req 7.1).

    Naming the dims explicitly stops PyImageJ from guessing axes from the shape, which
    could otherwise turn Z into channels.
    """
    _require_3d(arr)
    return get_ij().py.to_java(xr.DataArray(arr, dims=ZYX_DIMS))


def _as_zyx_numpy(converted) -> np.ndarray:
    """Return a detached numpy copy of a PyImageJ result, ordered (Z,Y,X) when dims are named.

    Singleton channel/time dims are dropped so a 1-channel, 1-frame stack reads back as 3D.
    The copy detaches the result from JVM-backed memory.
    """
    if isinstance(converted, xr.DataArray):
        extra_singletons = [dim for dim in converted.dims
                            if dim not in ZYX_DIMS and converted.sizes[dim] == 1]
        converted = converted.squeeze(extra_singletons)
        if set(converted.dims) == set(ZYX_DIMS):
            converted = converted.transpose(*ZYX_DIMS)
        return np.array(converted.values)
    return np.array(converted)


def from_java_array(java_image) -> np.ndarray:
    """Convert a Java image (Dataset, Img, RAI or ImagePlus) back to a numpy (Z,Y,X) array.

    Inverse of `to_java_img`: `from_java_array(to_java_img(a))` equals `a` (Req 7.2).
    """
    return _as_zyx_numpy(get_ij().py.from_java(java_image))


def _is_binary_zyx_stack(imp, shape: tuple[int, int, int]) -> bool:
    """True if an ImagePlus is 8-bit with Z slices of YxX, one channel and one frame (H4)."""
    z, y, x = shape
    return (int(imp.getBitDepth()) == 8 and int(imp.getNSlices()) == z
            and int(imp.getHeight()) == y and int(imp.getWidth()) == x
            and int(imp.getNChannels()) == 1 and int(imp.getNFrames()) == 1)


def _imageplus_via_xarray(voxels: np.ndarray):
    """Build an ImagePlus through PyImageJ's xarray route.

    `duplicate()` copies the pixels into a plain ImageStack, so the result does not depend
    on the numpy buffer staying alive and plugins see ordinary ByteProcessors.
    """
    return get_ij().py.to_imageplus(xr.DataArray(voxels, dims=ZYX_DIMS)).duplicate()


def _imageplus_via_stack(voxels: np.ndarray):
    """Build an ImagePlus slice by slice with ij.ImageStack (H4 fallback).

    ByteProcessor pixels are row-major (index = y*width + x), matching a C-ordered YxX slice.
    Java bytes are signed, so 255 is stored as -1 and read back by ImageJ as 255.
    """
    z, y, x = voxels.shape
    stack = scyjava.jimport("ij.ImageStack")(x, y)
    byte_processor_class = scyjava.jimport("ij.process.ByteProcessor")
    for index in range(z):
        pixels = scyjava.jarray("b", [y * x])
        pixels[:] = np.ascontiguousarray(voxels[index]).ravel().view(np.int8)
        stack.addSlice(f"z{index}", byte_processor_class(x, y, pixels))
    imp = scyjava.jimport("ij.ImagePlus")("mask", stack)
    imp.setDimensions(1, z, 1)
    return imp


def mask_to_imageplus(mask: np.ndarray):
    """Convert a boolean (Z,Y,X) mask to an uncalibrated 8-bit {0,255} ImagePlus (Req 7.3).

    The xarray route is tried first; if it does not yield Z slices x 1 channel x 1 frame
    (hypothesis H4 falsified), the stack is built explicitly instead.
    """
    _require_3d(mask)
    voxels = np.where(np.asarray(mask, dtype=bool), MASK_FOREGROUND, 0).astype(np.uint8)
    imp = _imageplus_via_xarray(voxels)
    if not _is_binary_zyx_stack(imp, voxels.shape):
        log.warning("xarray route gave %d slices/%d channels/%d frames for shape %s; "
                    "building the ImageStack explicitly (H4)", imp.getNSlices(),
                    imp.getNChannels(), imp.getNFrames(), voxels.shape)
        imp = _imageplus_via_stack(voxels)
    imp.setCalibration(scyjava.jimport("ij.measure.Calibration")())  # voxel units, not µm
    return imp


def _stack_voxels(imp) -> np.ndarray:
    """Read every stack slice of an ImagePlus into a numpy (Z,Y,X) array via its processors."""
    stack = imp.getStack()
    height, width = int(imp.getHeight()), int(imp.getWidth())
    slices = [np.array(stack.getProcessor(index).getPixels()).reshape(height, width)
              for index in range(1, int(stack.getSize()) + 1)]
    return np.stack(slices)


def imageplus_to_mask(imp) -> np.ndarray:
    """Convert an ImagePlus to a boolean (Z,Y,X) mask, True where the voxel is non-zero (Req 7.4).

    If PyImageJ's conversion does not give (stack size, height, width), the stack is read
    slice by slice instead, mirroring the H4 fallback of `mask_to_imageplus`.
    """
    expected_shape = (int(imp.getStackSize()), int(imp.getHeight()), int(imp.getWidth()))
    voxels = from_java_array(imp)
    if voxels.shape != expected_shape:
        log.warning("from_java gave shape %s, expected %s; reading the stack directly (H4)",
                    voxels.shape, expected_shape)
        voxels = _stack_voxels(imp)
    return voxels != 0


# --- Java tables -> pandas (Requirement 7.5) ---------------------------------------------


def _numeric_or_text(values) -> pd.Series:
    """Return values as a numeric Series when every entry parses as a number, else unchanged."""
    series = pd.Series(values)
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def _results_table_column(table, name: str, n_rows: int):
    """Read one ResultsTable column, preferring the full-precision double values.

    Text columns (e.g. "Label") have no numeric index or are all NaN as doubles; they are
    read as strings, since getStringValue rounds numbers to the table's decimal places.
    """
    index = int(table.getColumnIndex(name))
    doubles = table.getColumnAsDoubles(index) if index != _COLUMN_NOT_FOUND else None
    if doubles is not None:
        numeric = np.array(doubles, dtype=np.float64)
        if n_rows == 0 or not np.isnan(numeric).all():
            return numeric
    return _numeric_or_text([str(table.getStringValue(name, row)) for row in range(n_rows)])


def _results_table_columns(table) -> dict:
    """Return {heading: values} for an ij.measure.ResultsTable, in heading order."""
    n_rows = int(table.size())
    return {str(heading): _results_table_column(table, str(heading), n_rows)
            for heading in table.getHeadings()}


def _scijava_cell(cell):
    """Convert one SciJava table cell (boxed number, String or null) to a Python value."""
    return None if cell is None else scyjava.to_python(cell, gentle=True)


def _scijava_table_columns(table) -> dict:
    """Return {header: values} for an org.scijava.table.Table, in column order."""
    n_rows = int(table.getRowCount())
    return {str(table.getColumnHeader(col)):
            _numeric_or_text([_scijava_cell(table.get(col, row)) for row in range(n_rows)])
            for col in range(int(table.getColumnCount()))}


def table_to_dataframe(table) -> pd.DataFrame:
    """Convert an ij.measure.ResultsTable or org.scijava.table.Table to a DataFrame (Req 7.5).

    One row per table row, the Fiji column names unchanged, numeric dtypes for numeric
    columns. Renaming to internal names (COLUMN_MAP) is left to the caller.
    """
    if isinstance(table, scyjava.jimport("ij.measure.ResultsTable")):
        columns = _results_table_columns(table)
    elif isinstance(table, scyjava.jimport("org.scijava.table.Table")):
        columns = _scijava_table_columns(table)
    else:
        raise TypeError(f"unsupported table type: {type(table)!r}")
    return pd.DataFrame(columns)


# --- MorphoLibJ backend helpers (Requirement 6.13) ----------------------------------------
# The only place MORPHOLIBJ_BACKEND is read. Each helper returns a new Java object owned by the
# caller and creates no other Java images, so nothing per-volume outlives the call (Req 5.7).


def _require_direct_backend(operation: str) -> None:
    """Reject any backend other than "direct", the only one discovery verified headless."""
    if MORPHOLIBJ_BACKEND != "direct":
        raise RuntimeError(
            f"{operation}: MORPHOLIBJ_BACKEND={MORPHOLIBJ_BACKEND!r} is not implemented; only "
            "'direct' passed the headless probe. Run `python scripts/discover_fiji.py`."
        )


def label_components(imp):
    """Label the connected components of a binary 8-bit ImagePlus at CONNECTIVITY_3D.

    Returns a new 32-bit (float) label ImagePlus with labels 1..n and background 0.
    """
    _require_direct_backend("label_components")
    get_ij()
    binary_images = scyjava.jimport(CLASS_BINARY_IMAGES)
    return binary_images.componentsLabeling(imp, CONNECTIVITY_3D, LABEL_BIT_DEPTH)


def size_opening(labels, min_size: int):
    """Return a new label ImagePlus keeping only labels with at least `min_size` voxels."""
    _require_direct_backend("size_opening")
    get_ij()
    return scyjava.jimport(CLASS_LABEL_IMAGES).sizeOpening(labels, int(min_size))


def _morphometric_features():
    """Build MorphometricFeatures3D for MORPHOMETRIC_FEATURES_3D (default 13-direction Crofton)."""
    feature_enum = scyjava.jimport(f"{CLASS_ANALYZE_REGIONS_3D}$Feature")
    features = [feature_enum.valueOf(name) for name in MORPHOMETRIC_FEATURES_3D]
    return scyjava.jimport(CLASS_ANALYZE_REGIONS_3D)(features)


def _append_max_feret(table, labels) -> None:
    """Deprecated: the 3D max Feret is approximated from the equivalent ellipsoid (H7).

    MorphoLibJ's `MaxFeretDiameter3D.computeTable` scales as O(points²) per object and does not
    finish on real volumes with ~1800 components, so it is no longer called in the measurement
    path. `measure._feret_um` uses `2 × largest ellipsoid semi-axis` instead. Kept for reference;
    see docs/reproducibility.md under H7.
    """
    raise NotImplementedError("max Feret is approximated from the equivalent ellipsoid; see H7")


def analyze_regions_3d(labels):
    """Measure every label of an uncalibrated label ImagePlus; return an ij.measure.ResultsTable.

    Columns are MorphoLibJ's Analyze Regions 3D names (see COLUMN_MAP), all in voxel units.
    The 3D max Feret is intentionally not computed here: `MaxFeretDiameter3D` scales as
    O(points²) per object and does not finish on real volumes with ~1800 components. The
    max Feret is approximated downstream from the equivalent ellipsoid (measure._feret_um, H7).
    Scaling to µm happens once, in Python.
    """
    _require_direct_backend("analyze_regions_3d")
    get_ij()
    return _morphometric_features().computeTable(labels)
