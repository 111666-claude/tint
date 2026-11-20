package main

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ashish0kumar/tint/themes"
)

// testPalette returns a real theme palette for tests.
func testPalette(t *testing.T) []color.Color {
	t.Helper()
	palette, err := themes.GetPalette("catppuccin-mocha")
	if err != nil {
		t.Fatalf("failed to get test palette: %v", err)
	}
	return palette
}

// newUniformImage creates a w×h image filled with c, with bounds anchored at (0, 0).
func newUniformImage(w, h int, c color.Color) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

// newUniformMask creates a w×h mask filled with gray value v, anchored at (0, 0).
func newUniformMask(w, h int, v uint8) *image.Gray {
	mask := image.NewGray(image.Rect(0, 0, w, h))
	for i := range mask.Pix {
		mask.Pix[i] = v
	}
	return mask
}

// nrgbaAt returns the color at (x, y) converted to non-premultiplied form.
func nrgbaAt(img image.Image, x, y int) color.NRGBA {
	return color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
}

// --- Two-color clothing sample: mask selects only one garment ---

func TestMaskSelectsOnlyMaskedRegion(t *testing.T) {
	palette := testPalette(t)

	// "Photo": left half is a red shirt, right half is blue pants.
	shirt := color.NRGBA{R: 200, G: 30, B: 30, A: 255}
	pants := color.NRGBA{R: 30, G: 30, B: 200, A: 255}
	img := image.NewNRGBA(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			if x < 4 {
				img.Set(x, y, shirt)
			} else {
				img.Set(x, y, pants)
			}
		}
	}

	// Mask covers only the shirt (left half).
	mask := image.NewGray(image.Rect(0, 0, 8, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			mask.SetGray(x, y, color.Gray{Y: 255})
		}
	}

	masked := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, mask, 4, false)
	unmasked := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, nil, 4, false)

	if masked.Bounds() != img.Bounds() {
		t.Fatalf("output bounds %v != input bounds %v", masked.Bounds(), img.Bounds())
	}

	for y := 0; y < 4; y++ {
		// Inside the mask: must equal the fully themed result.
		gotIn := nrgbaAt(masked, 1, y)
		wantIn := nrgbaAt(unmasked, 1, y)
		if gotIn != wantIn {
			t.Errorf("inside mask at (1,%d): got %v, want themed %v", y, gotIn, wantIn)
		}
		if gotIn == shirt {
			t.Errorf("inside mask at (1,%d): shirt color %v was not recolored", y, gotIn)
		}

		// Outside the mask: must be exactly the original color.
		gotOut := nrgbaAt(masked, 6, y)
		if gotOut != pants {
			t.Errorf("outside mask at (6,%d): got %v, want original %v", y, gotOut, pants)
		}
	}
}

// --- Transparent edges: semi-transparent pixels must not gain black fringes ---

func TestMaskPreservesSemiTransparentEdges(t *testing.T) {
	palette := testPalette(t)

	// A soft edge: opaque white core fading to fully transparent.
	img := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 128})
	img.SetNRGBA(2, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 10})
	img.SetNRGBA(3, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 0})

	// Half-strength mask over the whole row.
	mask := newUniformMask(4, 1, 128)

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, mask, 1, false)

	// Alpha must be preserved exactly on every pixel.
	for x, wantA := range []uint8{255, 128, 10, 0} {
		if got := nrgbaAt(out, x, 0).A; got != wantA {
			t.Errorf("pixel %d: alpha = %d, want %d", x, got, wantA)
		}
	}

	// Fully transparent pixel stays transparent.
	if got := nrgbaAt(out, 3, 0); got != (color.NRGBA{}) {
		t.Errorf("transparent pixel: got %v, want fully transparent", got)
	}

	// The semi-transparent pixel was white; a half blend with any palette
	// color (catppuccin-mocha is a dark theme) must not darken it to black.
	// In premultiplied space a black fringe would show up as RGB << alpha.
	semi := nrgbaAt(out, 1, 0)
	if semi.R < 100 || semi.G < 100 || semi.B < 100 {
		t.Errorf("semi-transparent edge darkened towards black: got %v", semi)
	}
}

func TestMaskZeroWeightKeepsSemiTransparentPixelUntouched(t *testing.T) {
	palette := testPalette(t)

	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 250, G: 240, B: 230, A: 128})

	mask := newUniformMask(1, 1, 0) // weight 0 everywhere

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, mask, 1, false)
	got := nrgbaAt(out, 0, 0)

	// Allow ±1 rounding from the premultiplied round-trip through image.RGBA.
	if diff := int(got.R) - 250; diff < -1 || diff > 1 {
		t.Errorf("R = %d, want ~250 (no black fringe)", got.R)
	}
	if got.A != 128 {
		t.Errorf("A = %d, want 128", got.A)
	}
}

// --- Gradient mask: blend follows mask intensity ---

func TestGradientMaskBlendsLinearly(t *testing.T) {
	palette := testPalette(t)

	const w = 256
	orig := color.NRGBA{R: 200, G: 30, B: 30, A: 255}
	img := newUniformImage(w, 1, orig)

	// Horizontal gradient mask: weight = x/255.
	mask := image.NewGray(image.Rect(0, 0, w, 1))
	for x := 0; x < w; x++ {
		mask.SetGray(x, 0, color.Gray{Y: uint8(x)})
	}

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, mask, 1, false)
	full := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, nil, 1, false)

	// Left end (weight 0): exactly the original color.
	if got := nrgbaAt(out, 0, 0); got != orig {
		t.Errorf("gradient start: got %v, want original %v", got, orig)
	}

	// Right end (weight 1): equals the fully themed color.
	gotEnd := nrgbaAt(out, w-1, 0)
	wantEnd := nrgbaAt(full, 0, 0)
	if gotEnd != wantEnd {
		t.Errorf("gradient end: got %v, want themed %v", gotEnd, wantEnd)
	}

	// Middle: matches the documented blend formula.
	mid := nrgbaAt(out, 128, 0)
	weight := 128.0 / 255.0
	blend := func(o, t uint8) uint8 {
		return uint8(math.Round(float64(o)*(1-weight) + float64(t)*weight))
	}
	wantMid := color.NRGBA{
		R: blend(orig.R, wantEnd.R),
		G: blend(orig.G, wantEnd.G),
		B: blend(orig.B, wantEnd.B),
		A: 255,
	}
	if mid != wantMid {
		t.Errorf("gradient middle: got %v, want blended %v", mid, wantMid)
	}
}

// --- Mask with a different origin ---

func TestMaskWithDifferentOrigin(t *testing.T) {
	palette := testPalette(t)

	// Image anchored at a non-zero origin, like a sub-image of a larger canvas.
	imgBounds := image.Rect(10, 20, 14, 24)
	img := image.NewNRGBA(imgBounds)
	shirt := color.NRGBA{R: 200, G: 30, B: 30, A: 255}
	pants := color.NRGBA{R: 30, G: 30, B: 200, A: 255}
	for y := imgBounds.Min.Y; y < imgBounds.Max.Y; y++ {
		for x := imgBounds.Min.X; x < imgBounds.Max.X; x++ {
			if x < 12 {
				img.Set(x, y, shirt)
			} else {
				img.Set(x, y, pants)
			}
		}
	}

	// Mask anchored at yet another origin; left half selected.
	mask := image.NewGray(image.Rect(-3, 100, 1, 104))
	for y := 100; y < 104; y++ {
		mask.SetGray(-3, y, color.Gray{Y: 255})
		mask.SetGray(-2, y, color.Gray{Y: 255})
	}

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, mask, 2, false)

	if out.Bounds() != imgBounds {
		t.Fatalf("output bounds %v != input bounds %v", out.Bounds(), imgBounds)
	}

	// Top-left of the image is inside the mask -> recolored.
	if got := nrgbaAt(out, 10, 20); got == shirt {
		t.Errorf("masked pixel at image origin was not recolored: %v", got)
	}
	// Bottom-right is outside the mask -> original preserved.
	if got := nrgbaAt(out, 13, 23); got != pants {
		t.Errorf("unmasked pixel at (13,23): got %v, want original %v", got, pants)
	}
}

// --- Single-threaded and parallel runs must be pixel-identical ---

func TestSingleThreadedMatchesParallel(t *testing.T) {
	palette := testPalette(t)

	img := image.NewNRGBA(image.Rect(0, 0, 37, 23))
	for y := 0; y < 23; y++ {
		for x := 0; x < 37; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*7 + y*13) % 256),
				G: uint8((x*3 + y*5) % 256),
				B: uint8((x*11 + y) % 256),
				A: uint8(255 - (x+y)%3), // mix of opaque and semi-transparent
			})
		}
	}
	// Gradient mask.
	mask := image.NewGray(image.Rect(0, 0, 37, 23))
	for y := 0; y < 23; y++ {
		for x := 0; x < 37; x++ {
			mask.SetGray(x, y, color.Gray{Y: uint8((x*37 + y*91) % 256)})
		}
	}

	workers := runtime.NumCPU()
	for _, m := range []*image.Gray{nil, mask} {
		single := processImageWithWorkers(img, palette, 1.1, defaultNearest, defaultPower, m, 1, false)
		parallel := processImageWithWorkers(img, palette, 1.1, defaultNearest, defaultPower, m, workers, false)

		if len(single.Pix) != len(parallel.Pix) {
			t.Fatalf("mask=%v: buffer sizes differ", m != nil)
		}
		for i := range single.Pix {
			if single.Pix[i] != parallel.Pix[i] {
				t.Fatalf("mask=%v: pixel byte %d differs: single=%d parallel=%d",
					m != nil, i, single.Pix[i], parallel.Pix[i])
			}
		}
	}
}

// --- No mask: historical behavior is unchanged ---

func TestNilMaskPreservesOriginalBehavior(t *testing.T) {
	palette := testPalette(t)

	img := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 123, G: 45, B: 67, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 10, G: 20, B: 30, A: 0})    // transparent
	img.SetNRGBA(2, 0, color.NRGBA{R: 200, G: 100, B: 50, A: 90}) // semi-transparent

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, nil, 1, false)

	// Transparent pixel stays transparent.
	if got := nrgbaAt(out, 1, 0); got != (color.NRGBA{}) {
		t.Errorf("transparent pixel: got %v, want transparent", got)
	}

	// Opaque pixel matches a direct Shepard's-method computation (the
	// pre-mask pipeline), including the forced opaque alpha.
	paletteRGBAs := make([]color.RGBA, len(palette))
	for i, c := range palette {
		paletteRGBAs[i] = toRGBA(c)
	}
	want := toRGBA(shepardsMethodColor(color.RGBA{R: 123, G: 45, B: 67, A: 255}, paletteRGBAs, defaultNearest, defaultPower))
	if got := toRGBA(out.At(0, 0)); got != want {
		t.Errorf("opaque pixel: got %v, want %v", got, want)
	}
}

// --- Edge cases: empty mask, size mismatch, non-finite weights ---

func TestEmptyMaskKeepsImageUnchanged(t *testing.T) {
	palette := testPalette(t)

	orig := color.NRGBA{R: 77, G: 88, B: 99, A: 255}
	img := newUniformImage(5, 3, orig)

	// A zero-size mask: every lookup falls outside its bounds -> weight 0.
	empty := image.NewGray(image.Rectangle{})

	out := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, empty, 2, false)
	for y := 0; y < 3; y++ {
		for x := 0; x < 5; x++ {
			if got := nrgbaAt(out, x, y); got != orig {
				t.Fatalf("pixel (%d,%d): got %v, want original %v", x, y, got, orig)
			}
		}
	}
}

func TestClampMaskWeightHandlesNonFinite(t *testing.T) {
	cases := []struct {
		name string
		in   float64
		want float64
	}{
		{"NaN", math.NaN(), 0},
		{"+Inf", math.Inf(1), 0},
		{"-Inf", math.Inf(-1), 0},
		{"negative", -0.5, 0},
		{"zero", 0, 0},
		{"half", 0.5, 0.5},
		{"one", 1, 1},
		{"above one", 1.7, 1},
	}
	for _, tc := range cases {
		if got := clampMaskWeight(tc.in); got != tc.want {
			t.Errorf("%s: clampMaskWeight(%v) = %v, want %v", tc.name, tc.in, got, tc.want)
		}
	}

	// Non-finite weights must keep the original color in a blend.
	orig := color.NRGBA{R: 10, G: 20, B: 30, A: 200}
	themed := color.RGBA{R: 250, G: 240, B: 230, A: 255}
	for _, w := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got := blendTowardsTheme(orig, themed, w); got != orig {
			t.Errorf("blendTowardsTheme with w=%v: got %v, want original %v", w, got, orig)
		}
	}
}

func TestLoadMaskRejectsSizeMismatch(t *testing.T) {
	dir := t.TempDir()

	// 4x4 image, 2x2 mask.
	maskImg := newUniformMask(2, 2, 255)
	maskPath := filepath.Join(dir, "mask.png")
	f, err := os.Create(maskPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, maskImg); err != nil {
		t.Fatal(err)
	}
	f.Close()

	if _, err := loadMask(maskPath, image.Rect(0, 0, 4, 4)); err == nil {
		t.Fatal("expected size mismatch error, got nil")
	}
}

func TestLoadMaskAcceptsMatchingSize(t *testing.T) {
	dir := t.TempDir()

	maskImg := newUniformMask(4, 4, 200)
	maskPath := filepath.Join(dir, "mask.png")
	f, err := os.Create(maskPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, maskImg); err != nil {
		t.Fatal(err)
	}
	f.Close()

	mask, err := loadMask(maskPath, image.Rect(0, 0, 4, 4))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := mask.GrayAt(0, 0).Y; got != 200 {
		t.Errorf("mask value = %d, want 200", got)
	}
}

func TestLoadMaskRejectsMissingAndUndecodable(t *testing.T) {
	dir := t.TempDir()

	if _, err := loadMask(filepath.Join(dir, "does-not-exist.png"), image.Rect(0, 0, 4, 4)); err == nil {
		t.Error("expected error for missing mask file")
	}

	bogus := filepath.Join(dir, "bogus.png")
	if err := os.WriteFile(bogus, []byte("not an image"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadMask(bogus, image.Rect(0, 0, 4, 4)); err == nil {
		t.Error("expected error for undecodable mask file")
	}
}

// --- Very large images are rejected by validation ---

func TestDecodeRejectsOversizedImage(t *testing.T) {
	dir := t.TempDir()

	// One pixel over the dimension limit; cheap to encode.
	big := image.NewGray(image.Rect(0, 0, MaxImageDimension+1, 1))
	path := filepath.Join(dir, "big.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, big); err != nil {
		t.Fatal(err)
	}
	f.Close()

	if _, _, err := decodeAndValidateImage(path, "catppuccin-mocha", 1.0, defaultNearest, defaultPower); err == nil {
		t.Fatal("expected dimension limit error, got nil")
	}
}

func TestLoadMaskRejectsOversizedMask(t *testing.T) {
	dir := t.TempDir()

	big := image.NewGray(image.Rect(0, 0, MaxImageDimension+1, 1))
	path := filepath.Join(dir, "big-mask.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, big); err != nil {
		t.Fatal(err)
	}
	f.Close()

	if _, err := loadMask(path, image.Rect(0, 0, MaxImageDimension+1, 1)); err == nil {
		t.Fatal("expected dimension limit error, got nil")
	}
}

// --- Existing theme behavior still holds ---

func TestThemesPalettesValid(t *testing.T) {
	if err := themes.ValidateThemeData(); err != nil {
		t.Fatalf("theme data invalid: %v", err)
	}
	for _, name := range themes.GetAvailableThemeNames() {
		palette, err := themes.GetPalette(name)
		if err != nil {
			t.Errorf("GetPalette(%q): %v", name, err)
			continue
		}
		if len(palette) < 3 {
			t.Errorf("GetPalette(%q): only %d colors", name, len(palette))
		}
	}
}

func TestUnmaskedOutputMatchesMaskedFullWeight(t *testing.T) {
	palette := testPalette(t)

	// For opaque pixels, a full-strength mask must reproduce the unmasked result.
	img := newUniformImage(6, 6, color.NRGBA{R: 90, G: 140, B: 200, A: 255})
	full := newUniformMask(6, 6, 255)

	masked := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, full, 3, false)
	unmasked := processImageWithWorkers(img, palette, 1.0, defaultNearest, defaultPower, nil, 3, false)

	for i := range masked.Pix {
		if masked.Pix[i] != unmasked.Pix[i] {
			t.Fatalf("byte %d differs: masked=%d unmasked=%d", i, masked.Pix[i], unmasked.Pix[i])
		}
	}
}
