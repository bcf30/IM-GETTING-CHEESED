// batch_particle_analysis.ijm
// Fiji/ImageJ macro that mirrors the mozzavid Python segmentation chain (Requirement 22).
//
// For every float32 TIFF in an input directory it runs: Gaussian Blur 3D (sigma 1.5 in x/y/z),
// Otsu "dark stack" threshold + Convert to Mask (inverted to match pipeline.DEFAULT_INVERT=True),
// MorphoLibJ "Label Size Opening" (min size and connectivity matching mozzavid/fiji.py), then
// MorphoLibJ "Analyze Regions 3D". It saves one CSV of region measurements per volume.
//
// The command strings are the ones pinned in mozzavid/fiji.py so the macro and the Python code
// stay in agreement (CMD_SIZE_OPENING = "Label Size Opening", CMD_ANALYZE_REGIONS_3D =
// "Analyze Regions 3D", CONNECTIVITY_3D = 26).
//
// The input TIFFs are the float32 copies written by
//   python scripts/validate_segmentation.py --export-float32 DIR
// because ImageJ 1.x has no half-float type and cannot open the MozzaVID float16 TIFFs (H9).
//
// Run in the GUI:   Plugins > Macros > Run... and pick this file (edit the parameters below), or
// headless:         ImageJ --headless -macro macros/batch_particle_analysis.ijm "input,output,1.5,50,true"

// ---- Parameters -------------------------------------------------------------------------
inputDir = getDirectory("Choose the input directory of float32 TIFFs");
outputDir = getDirectory("Choose the output directory for CSV results");
sigma = 1.5;        // Gaussian sigma in voxels, matching preprocess.denoise
minSize = 50;       // size-opening threshold in voxels, matching segment.cleanup default
connectivity = 26;  // matches fiji.CONNECTIVITY_3D
invert = true;      // matches pipeline.DEFAULT_INVERT (fat inclusions are the darker phase)

// Allow headless override: -macro script.ijm "inputDir,outputDir,sigma,minSize,invert"
args = getArgument();
if (lengthOf(args) > 0) {
    parts = split(args, ",");
    inputDir = parts[0];
    outputDir = parts[1];
    sigma = parseFloat(parts[2]);
    minSize = parseInt(parts[3]);
    invert = (parts[4] == "true");
}

setBatchMode(true);
list = getFileList(inputDir);
for (i = 0; i < list.length; i++) {
    name = list[i];
    if (endsWith(name, ".tif") || endsWith(name, ".tiff")) {
        processVolume(inputDir + name, outputDir, name, sigma, minSize, connectivity, invert);
    }
}
setBatchMode(false);
print("done: processed " + list.length + " entries from " + inputDir);

// ---- Per-volume processing --------------------------------------------------------------
function processVolume(path, outputDir, name, sigma, minSize, connectivity, invert) {
    open(path);
    title = getTitle();
    run("Gaussian Blur 3D...", "x=" + sigma + " y=" + sigma + " z=" + sigma);
    // "dark" tells ImageJ the objects of interest are the dark pixels; "stack" uses the 3D histogram.
    setAutoThreshold("Otsu dark stack");
    maskOptions = "method=Otsu background=Dark black stack";
    if (invert) maskOptions = maskOptions + " create";
    run("Convert to Mask", maskOptions);
    labelled = sizeOpenAndLabel(minSize, connectivity);
    selectWindow(labelled);
    run("Analyze Regions 3D");
    saveAs("Results", outputDir + name + ".csv");
    run("Close All");
    if (isOpen("Results")) { selectWindow("Results"); run("Close"); }
}

// Connected-component labelling followed by MorphoLibJ size opening; returns the result title.
function sizeOpenAndLabel(minSize, connectivity) {
    run("Connected Components Labeling", "connectivity=" + connectivity + " type=[16 bits]");
    labelled = getTitle();
    run("Label Size Opening", "min=" + minSize);
    return getTitle();
}
