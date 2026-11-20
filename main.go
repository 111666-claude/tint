package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ashish0kumar/tint/themes"
)

const (
	MaxImageDimension = 10000    // Maximum allowed width or height of the image
	MaxImagePixels    = 50000000 // Maximum allowed number of pixels in the image (~7071x7071)

	// Default params for Shepard's Method
	defaultLuminosity = 1.0
	defaultNearest    = 30
	defaultPower      = 4.0

	// ANSI escape codes for formatting
	bold      = "\033[1m"
	underline = "\033[4m"
	reset     = "\033[0m"
)

var version = "dev"

// toRGBA converts any color.Color to color.RGBA
func toRGBA(c color.Color) color.RGBA {
	r, g, b, a := c.RGBA()
	return color.RGBA{R: uint8(r >> 8), G: uint8(g >> 8), B: uint8(b >> 8), A: uint8(a >> 8)}
}

// colorDistanceSquared calculates the squared euclidean distance between two colors in RGB space
func colorDistanceSquared(c1, c2 color.RGBA) float64 {
	dr := float64(c1.R) - float64(c2.R)
	dg := float64(c1.G) - float64(c2.G)
	db := float64(c1.B) - float64(c2.B)

	return dr*dr + dg*dg + db*db
}

// findNClosestColors finds the N closest colors in the given palette to the original color
// It returns a slice of structs containing the distance and the color, sorted by distance
func findNClosestColors(originalRGBA color.RGBA, paletteRGBAs []color.RGBA, n int) []struct {
	dist  float64
	color color.Color
} {
	if len(paletteRGBAs) == 0 {
		return nil
	}

	distances := make([]struct {
		dist  float64
		color color.Color
	}, 0, len(paletteRGBAs))

	for _, pRGBA := range paletteRGBAs {
		distances = append(distances, struct {
			dist  float64
			color color.Color
		}{dist: colorDistanceSquared(originalRGBA, pRGBA), color: pRGBA})
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].dist < distances[j].dist
	})

	if n > len(distances) {
		n = len(distances)
	}
	return distances[:n]
}

// blendColors takes a slice of colors and their corresponding weights and returns a single blended color
func blendColors(colors []color.Color, weights []float64) color.RGBA {
	if len(colors) == 0 || len(colors) != len(weights) {
		return color.RGBA{}
	}

	var sumR, sumG, sumB float64
	var totalWeight float64

	for i := range colors {
		rgba := toRGBA(colors[i])
		sumR += float64(rgba.R) * weights[i]
		sumG += float64(rgba.G) * weights[i]
		sumB += float64(rgba.B) * weights[i]
		totalWeight += weights[i]
	}

	if totalWeight == 0 {
		return toRGBA(colors[0]) // Fallback to the first color if weights are somehow zero
	}

	return color.RGBA{
		R: uint8(math.Round(sumR / totalWeight)),
		G: uint8(math.Round(sumG / totalWeight)),
		B: uint8(math.Round(sumB / totalWeight)),
		A: 255,
	}
}

// applyLuminosity adjusts a color's brightness by scaling its RGB components
func applyLuminosity(c color.RGBA, factor float64) color.RGBA {
	r := uint8(math.Max(0, math.Min(255, float64(c.R)*factor)))
	g := uint8(math.Max(0, math.Min(255, float64(c.G)*factor)))
	b := uint8(math.Max(0, math.Min(255, float64(c.B)*factor)))
	return color.RGBA{R: r, G: g, B: b, A: c.A}
}

// shepardsMethodColor applies Shepard's Method for color interpolation
// It finds the 'nearest' palette colors and blends them using inverse distance weighting
func shepardsMethodColor(originalRGBA color.RGBA, paletteRGBAs []color.RGBA, nearest int, power float64) color.Color {
	closest := findNClosestColors(originalRGBA, paletteRGBAs, nearest)
	if len(closest) == 0 {
		return originalRGBA // No palette colors available, return original color
	}
	// If an exact match is found or only one neighbor is requested, just return it
	if len(closest) == 1 || closest[0].dist == 0 {
		return closest[0].color
	}

	weights := make([]float64, len(closest))
	var totalWeight float64
	for i, c := range closest {
		if c.dist == 0 { // Avoid division by zero
			return c.color
		}
		// Inverse distance weighting
		weight := 1.0 / math.Pow(math.Sqrt(c.dist), power)
		weights[i] = weight
		totalWeight += weight
	}

	if totalWeight == 0 { // Fallback if all weights somehow sum to zero
		return closest[0].color
	}

	return blendColors(extractColors(closest), weights)
}

// extractColors pulls just the color.Color from the sorted slice of (distance, color) tuples
func extractColors(sortedColors []struct {
	dist  float64
	color color.Color
}) []color.Color {
	colors := make([]color.Color, len(sortedColors))
	for i, item := range sortedColors {
		colors[i] = item.color
	}
	return colors
}

// ProgressTracker tracks and displays processing progress
type ProgressTracker struct {
	total       int64
	processed   int64
	startTime   time.Time
	lastUpdate  time.Time
	updateMutex sync.Mutex
}

// NewProgressTracker creates a new progress tracker
func NewProgressTracker(total int64) *ProgressTracker {
	return &ProgressTracker{
		total:      total,
		startTime:  time.Now(),
		lastUpdate: time.Now(),
	}
}

// updateProgress increments the processed count and displays progress
func (pt *ProgressTracker) updateProgress(increment int64) {
	atomic.AddInt64(&pt.processed, increment)

	pt.updateMutex.Lock()
	defer pt.updateMutex.Unlock()

	now := time.Now()
	if now.Sub(pt.lastUpdate) < 100*time.Millisecond {
		return
	}
	pt.lastUpdate = now

	processed := atomic.LoadInt64(&pt.processed)
	if processed >= pt.total {
		return
	}

	percentage := float64(processed) / float64(pt.total) * 100
	elapsed := now.Sub(pt.startTime)

	if processed > 0 {
		estimatedTotal := time.Duration(float64(elapsed) / float64(processed) * float64(pt.total))
		remaining := estimatedTotal - elapsed

		fmt.Printf("\rProgress: %.1f%% (%d/%d) Elapsed: %v ETA: %v",
			percentage, processed, pt.total, elapsed.Round(time.Second), remaining.Round(time.Second))
	}
}

// finishProgress completes the progress display
func (pt *ProgressTracker) finishProgress() {
	processed := atomic.LoadInt64(&pt.processed)
	elapsed := time.Since(pt.startTime)
	fmt.Printf("\rComplete: 100.0%% (%d/%d) in %v\n",
		processed, pt.total, elapsed.Round(time.Millisecond))
}

// clampMaskWeight normalizes a mask weight to [0, 1].
// Non-finite weights (NaN, ±Inf) and negative weights are treated as 0
// (keep the original color); weights above 1 are clamped to 1 (full theme color).
func clampMaskWeight(w float64) float64 {
	if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
		return 0
	}
	if w > 1 {
		return 1
	}
	return w
}

// maskWeightAt returns the mask weight in [0, 1] for the image pixel at (x, y).
// The mask may have a different origin than the image; coordinates are
// translated so that the mask's top-left aligns with the image's top-left.
// Pixels outside the mask bounds yield weight 0 (original color kept).
func maskWeightAt(mask *image.Gray, imgBounds image.Rectangle, x, y int) float64 {
	mb := mask.Bounds()
	mx := x - imgBounds.Min.X + mb.Min.X
	my := y - imgBounds.Min.Y + mb.Min.Y
	return clampMaskWeight(float64(mask.GrayAt(mx, my).Y) / 255.0)
}

// blendTowardsTheme mixes the original (non-premultiplied) color with the
// themed color according to the mask weight w in [0, 1]:
// w == 0 keeps the original color, w == 1 gives the full themed color.
// The original alpha is always preserved.
func blendTowardsTheme(orig color.NRGBA, themed color.RGBA, w float64) color.NRGBA {
	w = clampMaskWeight(w)
	if w == 0 {
		return orig
	}
	blend := func(o, t uint8) uint8 {
		return uint8(math.Round(float64(o)*(1-w) + float64(t)*w))
	}
	return color.NRGBA{
		R: blend(orig.R, themed.R),
		G: blend(orig.G, themed.G),
		B: blend(orig.B, themed.B),
		A: orig.A,
	}
}

// recolorPixel computes the output color for a single pixel. It is the only
// place per-pixel decisions are made, so single-threaded and parallel runs
// produce identical results.
//
// Without a mask the historical behavior is preserved exactly. With a mask,
// colors are computed in non-premultiplied (NRGBA) space so semi-transparent
// edge pixels keep both their color and alpha instead of darkening to black.
func recolorPixel(
	img image.Image,
	mask *image.Gray,
	paletteRGBAs []color.RGBA,
	luminosity float64,
	nearest int,
	power float64,
	bounds image.Rectangle,
	x, y int,
) color.Color {
	if mask == nil {
		originalRGBA := toRGBA(img.At(x, y))
		if originalRGBA.A == 0 {
			return color.Transparent
		}
		adjustedColor := applyLuminosity(originalRGBA, luminosity)
		return shepardsMethodColor(adjustedColor, paletteRGBAs, nearest, power)
	}

	orig := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
	if orig.A == 0 {
		return color.Transparent
	}

	adjustedColor := applyLuminosity(color.RGBA{R: orig.R, G: orig.G, B: orig.B, A: 255}, luminosity)
	themed := toRGBA(shepardsMethodColor(adjustedColor, paletteRGBAs, nearest, power))

	w := maskWeightAt(mask, bounds, x, y)
	return blendTowardsTheme(orig, themed, w)
}

// processRows recolors the row range [startY, endY) of img into dst.
func processRows(
	img image.Image,
	mask *image.Gray,
	paletteRGBAs []color.RGBA,
	luminosity float64,
	nearest int,
	power float64,
	bounds image.Rectangle,
	startY, endY int,
	dst *image.RGBA,
	progress *ProgressTracker,
) {
	pixelsProcessed := int64(0)
	for y := startY; y < endY; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			dst.Set(x, y, recolorPixel(img, mask, paletteRGBAs, luminosity, nearest, power, bounds, x, y))
			pixelsProcessed++
		}

		// Update progress every 10 rows or at last row
		if progress != nil && ((y-startY)%10 == 0 || y == endY-1) {
			progress.updateProgress(pixelsProcessed)
			pixelsProcessed = 0
		}
	}
	if progress != nil && pixelsProcessed > 0 {
		progress.updateProgress(pixelsProcessed)
	}
}

// processImageWithShepardsMethod applies Shepard's Method to each pixel of the
// image concurrently. If mask is non-nil, its gray value per pixel controls
// the blend between the original color and the themed color.
func processImageWithShepardsMethod(
	img image.Image,
	palette []color.Color,
	luminosity float64,
	nearest int,
	power float64,
	mask *image.Gray,
) *image.RGBA {
	return processImageWithWorkers(img, palette, luminosity, nearest, power, mask, runtime.NumCPU(), true)
}

// processImageWithWorkers is the testable core of processImageWithShepardsMethod:
// numWorkers controls parallelism and showProgress toggles the progress display.
// The result is independent of numWorkers because each worker writes a disjoint
// row range through the shared recolorPixel logic.
func processImageWithWorkers(
	img image.Image,
	palette []color.Color,
	luminosity float64,
	nearest int,
	power float64,
	mask *image.Gray,
	numWorkers int,
	showProgress bool,
) *image.RGBA {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// Pre-convert palette colors to RGBA once
	paletteRGBAs := make([]color.RGBA, len(palette))
	for i, c := range palette {
		paletteRGBAs[i] = toRGBA(c)
	}

	// Initialize progress tracker
	var progress *ProgressTracker
	if showProgress {
		totalPixels := int64(width * height)
		progress = NewProgressTracker(totalPixels)
	}

	// Determine number of workers based on CPU cores
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > height {
		numWorkers = height // no more workers than rows
	}

	// Calculate rows per worker with ceiling division
	rowsPerWorker := (height + numWorkers - 1) / numWorkers

	// Prepare a slice to hold partial images from workers
	partialImages := make([]*image.RGBA, numWorkers)

	var wg sync.WaitGroup
	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Determine row range for this worker
			startY := bounds.Min.Y + id*rowsPerWorker
			endY := startY + rowsPerWorker
			if endY > bounds.Max.Y {
				endY = bounds.Max.Y
			}

			// Create partial image buffer for this worker
			partialImg := image.NewRGBA(image.Rect(bounds.Min.X, startY, bounds.Max.X, endY))

			processRows(img, mask, paletteRGBAs, luminosity, nearest, power, bounds, startY, endY, partialImg, progress)

			partialImages[id] = partialImg
		}(workerID)
	}

	wg.Wait()
	if progress != nil {
		progress.finishProgress()
	}

	// Merge partial images into final image
	newImg := image.NewRGBA(bounds)
	for _, pImg := range partialImages {
		if pImg == nil {
			continue
		}
		for y := pImg.Bounds().Min.Y; y < pImg.Bounds().Max.Y; y++ {
			for x := pImg.Bounds().Min.X; x < pImg.Bounds().Max.X; x++ {
				newImg.Set(x, y, pImg.At(x, y))
			}
		}
	}

	return newImg
}

// decodeAndValidateImage opens, decodes, and validates the image.
func decodeAndValidateImage(imagePath string, themeAndFlavor string, luminosity float64, nearest int, power float64) (image.Image, string, error) {
	// Open file
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, "", fmt.Errorf("cannot open image file '%s': %v", imagePath, err)
	}
	defer file.Close()

	// Get file info
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, "", fmt.Errorf("cannot get file info for '%s': %v", imagePath, err)
	}

	// Check file size
	if fileInfo.Size() > 100*1024*1024 {
		return nil, "", fmt.Errorf("image file '%s' is too large (%.2f MB). Maximum size is 100 MB",
			imagePath, float64(fileInfo.Size())/(1024*1024))
	}

	// Decode image
	img, format, err := image.Decode(file)
	if err != nil {
		return nil, "", fmt.Errorf("cannot decode image '%s': %v. Must be a valid JPEG or PNG", imagePath, err)
	}
	if format != "jpeg" && format != "png" {
		return nil, "", fmt.Errorf("unsupported image format '%s'. Only JPEG and PNG are supported", format)
	}

	// Check dimensions
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width > MaxImageDimension || height > MaxImageDimension {
		return nil, "", fmt.Errorf("image dimensions too large (%dx%d). Max is %d", width, height, MaxImageDimension)
	}
	if width*height > MaxImagePixels {
		return nil, "", fmt.Errorf("image has too many pixels (%d). Max is %d", width*height, MaxImagePixels)
	}

	// Validate theme
	if _, err := themes.GetPalette(themeAndFlavor); err != nil {
		return nil, "", fmt.Errorf("theme validation failed: %v", err)
	}

	// Validate parameters
	if luminosity <= 0 {
		return nil, "", fmt.Errorf("luminosity must be positive, got %.2f", luminosity)
	}
	if nearest < 1 {
		return nil, "", fmt.Errorf("nearest colors count must be at least 1, got %d", nearest)
	}
	if power <= 0 {
		return nil, "", fmt.Errorf("power must be positive, got %.2f", power)
	}

	return img, format, nil
}

// loadMask opens, decodes, and validates a grayscale mask image. The mask's
// gray value per pixel controls how strongly the theme color is applied
// (0 = keep original, 255 = full theme color). The mask must have the same
// width and height as the input image, but may have a different origin.
func loadMask(maskPath string, imgBounds image.Rectangle) (*image.Gray, error) {
	// Open file
	file, err := os.Open(maskPath)
	if err != nil {
		return nil, fmt.Errorf("cannot open mask file '%s': %v", maskPath, err)
	}
	defer file.Close()

	// Get file info
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("cannot get file info for mask '%s': %v", maskPath, err)
	}

	// Check file size
	if fileInfo.Size() > 100*1024*1024 {
		return nil, fmt.Errorf("mask file '%s' is too large (%.2f MB). Maximum size is 100 MB",
			maskPath, float64(fileInfo.Size())/(1024*1024))
	}

	// Decode mask
	maskImg, format, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("cannot decode mask '%s': %v. Must be a valid JPEG or PNG", maskPath, err)
	}
	if format != "jpeg" && format != "png" {
		return nil, fmt.Errorf("unsupported mask format '%s'. Only JPEG and PNG are supported", format)
	}

	// Check dimensions
	mb := maskImg.Bounds()
	if mb.Dx() == 0 || mb.Dy() == 0 {
		return nil, fmt.Errorf("mask '%s' is empty (%dx%d)", maskPath, mb.Dx(), mb.Dy())
	}
	if mb.Dx() > MaxImageDimension || mb.Dy() > MaxImageDimension {
		return nil, fmt.Errorf("mask dimensions too large (%dx%d). Max is %d", mb.Dx(), mb.Dy(), MaxImageDimension)
	}
	if mb.Dx()*mb.Dy() > MaxImagePixels {
		return nil, fmt.Errorf("mask has too many pixels (%d). Max is %d", mb.Dx()*mb.Dy(), MaxImagePixels)
	}

	// The mask must cover the image exactly (origin may differ)
	if mb.Dx() != imgBounds.Dx() || mb.Dy() != imgBounds.Dy() {
		return nil, fmt.Errorf("mask size (%dx%d) does not match image size (%dx%d)",
			mb.Dx(), mb.Dy(), imgBounds.Dx(), imgBounds.Dy())
	}

	// Convert to grayscale, preserving the mask's own bounds/origin
	gray := image.NewGray(mb)
	draw.Draw(gray, mb, maskImg, mb.Min, draw.Src)

	return gray, nil
}

// getOutputExtension determines the output file extension based on input format and output path
func getOutputExtension(inputFormat string, outputPath string) string {
	if outputPath != "" {
		// If output path is specified, use its extension
		ext := strings.ToLower(filepath.Ext(outputPath))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			return ext
		}
	}

	// Preserve input format
	switch inputFormat {
	case "jpeg":
		return ".jpg"
	case "png":
		return ".png"
	default:
		return ".png" // Fallback
	}
}

// saveImage saves the processed image
func saveImage(img image.Image, outputPath string, inputFormat string) error {
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("error creating output file '%s': %v", outputPath, err)
	}
	defer outFile.Close()

	outputExt := getOutputExtension(inputFormat, outputPath)

	switch outputExt {
	case ".jpg", ".jpeg":
		var opt jpeg.Options
		opt.Quality = 85
		if err := jpeg.Encode(outFile, img, &opt); err != nil {
			return fmt.Errorf("error encoding JPEG image to '%s': %v", outputPath, err)
		}
	case ".png":
		if err := png.Encode(outFile, img); err != nil {
			return fmt.Errorf("error encoding PNG image to '%s': %v", outputPath, err)
		}
	default:
		return fmt.Errorf("unsupported output format '%s'. Use .png, .jpg, or .jpeg", outputExt)
	}

	return nil
}

// generateOutputPath creates the output path based on input path, theme and format
func generateOutputPath(inputPath string, themeAndFlavor string, inputFormat string) string {
	dir := filepath.Dir(inputPath)
	base := filepath.Base(inputPath)
	nameWithoutExt := strings.TrimSuffix(base, filepath.Ext(base))

	ext := getOutputExtension(inputFormat, "")

	return filepath.Join(dir, fmt.Sprintf("%s_themed_%s%s",
		nameWithoutExt, strings.ToLower(themeAndFlavor), ext))
}

// listThemes prints all available themes and their flavors
func listThemes() {
	programName := filepath.Base(os.Args[0])

	fmt.Printf("\n%s%sUsage:%s %s --image <IMAGE> --theme <THEME-FLAVOR> [OPTIONS]\n\n", bold, underline, reset, programName)
	fmt.Printf("%s%sAvailable Themes & Flavors:%s\n", bold, underline, reset)

	themeNames := themes.GetAvailableThemeNames()

	for _, themeName := range themeNames {
		flavors := themes.GetAvailableFlavorNames(themeName)
		fmt.Printf("\n  %s%s%s\n", bold, themeName, reset)

		if len(flavors) > 0 {
			for _, flavor := range flavors {
				fmt.Printf("    - %s-%s\n", themeName, flavor)
			}
		}
	}

	fmt.Println()
}

// openFileInDefaultViewer attempts to open a file using the OS default viewer.
func openFileInDefaultViewer(filePath string) {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", filePath)

	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", filePath)

	case "linux":
		if _, err := exec.LookPath("xdg-open"); err != nil {
			log.Println("xdg-open not found; please install it to open files.")
			return
		}
		cmd = exec.Command("xdg-open", filePath)

	default:
		log.Printf("Unsupported operating system: %s", runtime.GOOS)
		return
	}

	if err := cmd.Run(); err != nil {
		log.Printf("Failed to open file '%s': %v", filePath, err)
	} else {
		log.Printf("Opened '%s' successfully.", filePath)
	}
}

func main() {
	log.SetFlags(0)
	themes.ValidateThemeData() // Validate theme data at startup

	// --- Define variables for flags ---

	var imagePath string
	var themeAndFlavor string
	var outputPath string
	var maskPath string
	var luminosity float64
	var nearest int
	var power float64
	var listThemesFlag bool
	var showVersion bool
	var open bool

	// --- Define and parse flags ---

	flag.StringVar(&imagePath, "image", "", "Path to the input image (required). Supports JPEG, PNG")
	flag.StringVar(&imagePath, "i", "", "Shorthand for -image")

	flag.StringVar(&themeAndFlavor, "theme", "", "Theme palette and optional flavor. Use -list-themes or -l to see all options.")
	flag.StringVar(&themeAndFlavor, "t", "", "Shorthand for -theme")

	flag.StringVar(&outputPath, "output", "", "Path for the output image (default: <input_filename>_themed_<theme-flavor>.<input_format>)")
	flag.StringVar(&outputPath, "o", "", "Shorthand for -output")

	flag.StringVar(&maskPath, "mask", "", "Path to a grayscale mask image (optional). Mask intensity blends the original color with the themed color (black = keep original, white = full theme). Must match the image size.")
	flag.StringVar(&maskPath, "m", "", "Shorthand for -mask")

	flag.BoolVar(&listThemesFlag, "list-themes", false, "List all available themes and their flavors")
	flag.BoolVar(&listThemesFlag, "l", false, "Shorthand for -list-themes")

	flag.BoolVar(&showVersion, "version", false, "Print the program version and exit")
	flag.BoolVar(&showVersion, "v", false, "Shorthand for -version")

	flag.BoolVar(&open, "not-open", false, "not open the recolored image in the default viewer")

	// Params specific to Shepard's Method
	flag.Float64Var(&luminosity, "luminosity", defaultLuminosity, "Luminosity adjustment factor (e.g., 0.8 for darker, 1.2 for brighter)")
	flag.IntVar(&nearest, "nearest", defaultNearest, "Number of nearest palette colors to consider for interpolation")
	flag.Float64Var(&power, "power", defaultPower, "Power for Shepard's Method (influences how quickly weights fall off)")

	flag.Usage = setUsage

	flag.Parse()

	// --- Handle version flag ---
	if showVersion {
		fmt.Printf("%s version %s\n", filepath.Base(os.Args[0]), version)
		os.Exit(0)
	}

	// --- Handle listThemesFlag ---
	if listThemesFlag {
		listThemes()
		os.Exit(0)
	}

	// --- Validate required args ---

	if imagePath == "" {
		fmt.Fprintln(os.Stderr, "Error: -i or --image <IMAGE> is required.")
		flag.Usage()
		os.Exit(1)
	}

	if themeAndFlavor == "" {
		fmt.Fprintln(os.Stderr, "Error: -t or --theme <THEME-FLAVOR> is required.")
		flag.Usage()
		os.Exit(1)
	}

	// --- Decode and validate image ---
	img, format, err := decodeAndValidateImage(imagePath, themeAndFlavor, luminosity, nearest, power)
	if err != nil {
		log.Fatalf("Validation failed: %v", err)
	}

	// --- Get palette ---
	paletteColors, err := themes.GetPalette(themeAndFlavor)
	if err != nil {
		log.Fatalf("Error getting palette: %v", err)
	}

	// --- Load optional mask ---
	var mask *image.Gray
	if maskPath != "" {
		mask, err = loadMask(maskPath, img.Bounds())
		if err != nil {
			log.Fatalf("Mask validation failed: %v", err)
		}
	}

	// --- Load image ---
	// (Already loaded above)

	// --- Process image with shepard's method ---
	log.Printf("Theme: %s", strings.ToLower(themeAndFlavor))
	log.Printf("Shepard's Method: nearest = %d, power = %.1f, luminosity = %.1f", nearest, power, luminosity)
	log.Printf("Processing: '%s'", imagePath)
	if mask != nil {
		log.Printf("Mask: '%s'", maskPath)
	}

	processedImg := processImageWithShepardsMethod(img, paletteColors, luminosity, nearest, power, mask)

	// --- Determine output path ---
	outPath := outputPath
	if outPath == "" {
		outPath = generateOutputPath(imagePath, themeAndFlavor, format)
	}

	// --- Save image ---
	if err := saveImage(processedImg, outPath, format); err != nil {
		log.Fatalf("Failed to save image: %v", err)
	}

	log.Printf("Saved image: '%s'\n", outPath)

	// --- Open output image in default viewer ---
	if !open {
		log.Print(outPath)
		openFileInDefaultViewer(outPath)
	}
}

// setUsage prints the custom help message
func setUsage() {
	programName := filepath.Base(os.Args[0])
	w := flag.CommandLine.Output()

	// Usage
	fmt.Fprintf(w, "\n%s%sUsage:%s %s --image <IMAGE> --theme <THEME-FLAVOR> [OPTIONS]\n\n", bold, underline, reset, programName)

	// Theme
	fmt.Fprintf(w, "  %s--theme, -t <STRING>%s\n", bold, reset)
	fmt.Fprintf(w, "\tTheme palette and optional flavor (required).\n")
	fmt.Fprintf(w, "\tUse --list-themes to see all available themes and flavors.\n\n")

	// Image
	fmt.Fprintf(w, "  %s--image, -i <PATH>%s\n", bold, reset)
	fmt.Fprintf(w, "\tPath to the input image (required). Supports JPEG, PNG formats.\n\n")

	// Options heading
	fmt.Fprintf(w, "%s%sOptions:%s\n\n", bold, underline, reset)

	// Output
	fmt.Fprintf(w, "  %s--output, -o <PATH>%s\n", bold, reset)
	fmt.Fprintf(w, "\tPath for the output image.\n")
	fmt.Fprintf(w, "\t(Default: <input_filename>_themed_<theme-flavor>.<input_format>)\n\n")

	// Mask
	fmt.Fprintf(w, "  %s--mask, -m <PATH>%s\n", bold, reset)
	fmt.Fprintf(w, "\tPath to a grayscale mask image (optional).\n")
	fmt.Fprintf(w, "\tMask intensity blends the original color with the themed color:\n")
	fmt.Fprintf(w, "\tblack keeps the original pixel, white applies the full theme color.\n")
	fmt.Fprintf(w, "\tUseful to recolor only selected regions (e.g. clothing in a photo)\n")
	fmt.Fprintf(w, "\twhile leaving skin and background untouched. Must match the image size.\n\n")

	// Luminosity
	fmt.Fprintf(w, "  %s--luminosity <FLOAT>%s\n", bold, reset)
	fmt.Fprintf(w, "\tLuminosity adjustment factor (e.g., 0.8 for darker, 1.2 for brighter).\n")
	fmt.Fprintf(w, "\t(Default: %.1f)\n\n", defaultLuminosity)

	// Nearest
	fmt.Fprintf(w, "  %s--nearest <COUNT>%s\n", bold, reset)
	fmt.Fprintf(w, "\tNumber of nearest palette colors to consider for interpolation.\n")
	fmt.Fprintf(w, "\t(Default: %d)\n\n", defaultNearest)

	// Power
	fmt.Fprintf(w, "  %s--power <FLOAT>%s\n", bold, reset)
	fmt.Fprintf(w, "\tPower for Shepard's Method (influences how quickly weights fall off).\n")
	fmt.Fprintf(w, "\t(Default: %.1f)\n\n", defaultPower)

	// List Themes
	fmt.Fprintf(w, "  %s--list-themes, -l%s\n", bold, reset)
	fmt.Fprintf(w, "\tList all available themes and their flavors.\n\n")

	// Open Option
	fmt.Fprintf(w, "  %s--not-open\n%s", bold, reset)
	fmt.Fprintf(w, "\tnot open the recolored image in the default viewer \n\n")

	// Version
	fmt.Fprintf(w, "  %s--version, -v%s\n", bold, reset)
	fmt.Fprintf(w, "\tPrint the program version and exit.\n\n")

	// Help
	fmt.Fprintf(w, "  %s--help, -h%s\n", bold, reset)
	fmt.Fprintf(w, "\tPrint this help message.\n\n")

	// Memory Considerations
	fmt.Fprintf(w, "%s%sMemory Note:%s\n\n", bold, underline, reset)
	fmt.Fprintf(w, "  Processing large images (e.g., %dMP, ~%dx%d) can use significant RAM.\n", MaxImagePixels/1000000, int(math.Sqrt(float64(MaxImagePixels))), int(math.Sqrt(float64(MaxImagePixels))))
	fmt.Fprintf(w, "  A %dMP image in RGBA format (4 bytes/pixel) may consume %sover 500 MiB of memory%s.\n", MaxImagePixels/1000000, bold, reset)
	fmt.Fprintf(w, "  Ensure your system has enough free memory before running.\n\n")
}
