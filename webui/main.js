/* ============================================================
   TRELLIS.2 WebUI — Main Application Logic
   ============================================================ */

// -------- State --------
const state = {
  activeTab: 'single',
  singleImage: null,       // File
  multiviewImages: [null, null, null, null, null, null],
  meshFile: null,
  texImage: null,
  sessionId: null,
  previews: null,           // { render_key: [base64...] }
  currentRenderMode: 'shaded_forest',
  currentAngle: 3,
  glbUrl: null,
  generating: false,
  extracting: false,
  currentPreset: 'balanced',
  currentSurface: 'hard',
};

// -------- Preset Definitions --------
// Each preset defines base values; surface type adds guidance offsets.
const PRESETS = {
  fast: {
    ss_gs: 7.5, ss_gr: 0.7, ss_steps: 12, ss_rt: 5.0,
    shape_gs: 7.5, shape_gr: 0.5, shape_steps: 12, shape_rt: 3.0,
    tex_gs: 1.0, tex_gr: 0.0, tex_steps: 12, tex_rt: 3.0,
    decimation: 500000, texSize: 2048,
  },
  balanced: {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 20, ss_rt: 5.0,
    shape_gs: 8.0, shape_gr: 0.5, shape_steps: 20, shape_rt: 3.0,
    tex_gs: 2.0, tex_gr: 0.1, tex_steps: 20, tex_rt: 3.0,
    decimation: 800000, texSize: 2048,
  },
  high: {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 30, ss_rt: 5.0,
    shape_gs: 8.5, shape_gr: 0.5, shape_steps: 30, shape_rt: 4.0,
    tex_gs: 2.5, tex_gr: 0.15, tex_steps: 30, tex_rt: 4.0,
    decimation: 800000, texSize: 4096,
  },
  max: {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 50, ss_rt: 6.0,
    shape_gs: 8.5, shape_gr: 0.5, shape_steps: 50, shape_rt: 6.0,
    tex_gs: 2.5, tex_gr: 0.2, tex_steps: 50, tex_rt: 6.0,
    decimation: 1000000, texSize: 4096,
  },
};

// Surface type guidance offsets (added to preset base)
const SURFACE_OFFSETS = {
  hard:    { ss_gs: +0.5, shape_gs: +0.5, tex_gs: 0 },   // higher guidance for crisp edges
  organic: { ss_gs: -0.5, shape_gs: -1.0, tex_gs: +0.5 }, // softer shape, richer textures
};

// -------- DOM Refs --------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// -------- Init --------
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSingleUpload();
  initMultiViewUpload();
  initTexturingUpload();
  initRadioGroups();
  initSliders();
  initPresets();
  initSurfaceType();
  initGenerateButton();
  initViewerModes();
  initRenderModes();
  initAngleSlider();
  initExtractButton();
  loadExamples();
  applyPreset(); // Set initial slider values from default preset
});

// ============================================================
// TABS
// ============================================================
function initTabs() {
  $$('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;
      $$('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      $$('.tab-content').forEach((c) => {
        c.classList.toggle('active', c.dataset.tab === tab);
      });
    });
  });
}

// ============================================================
// SINGLE IMAGE UPLOAD
// ============================================================
function initSingleUpload() {
  const zone = $('#singleUpload');
  const input = $('#singleFileInput');
  const preview = $('#singlePreview');
  const placeholder = $('#singlePlaceholder');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleSingleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => {
    if (input.files.length) handleSingleFile(input.files[0]);
  });

  function handleSingleFile(file) {
    state.singleImage = file;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    zone.classList.add('has-image');
  }
}

// ============================================================
// MULTI-VIEW UPLOAD
// ============================================================
function initMultiViewUpload() {
  const slots = $$('.mv-slot');
  slots.forEach((slot) => {
    const idx = parseInt(slot.dataset.index);
    const input = slot.querySelector('input[type=file]');
    const preview = slot.querySelector('.mv-preview');

    slot.addEventListener('click', () => input.click());
    slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('dragover'); });
    slot.addEventListener('dragleave', () => slot.classList.remove('dragover'));
    slot.addEventListener('drop', (e) => {
      e.preventDefault();
      slot.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleMvFile(e.dataTransfer.files[0], idx, slot, preview);
    });
    input.addEventListener('change', () => {
      if (input.files.length) handleMvFile(input.files[0], idx, slot, preview);
    });
  });

  function handleMvFile(file, idx, slot, preview) {
    state.multiviewImages[idx] = file;
    preview.src = URL.createObjectURL(file);
    slot.classList.add('has-image');
    updateMvCount();
  }
}

function updateMvCount() {
  const count = state.multiviewImages.filter((f) => f !== null).length;
  $('#mvCount').textContent = `${count} / 6`;
}

// ============================================================
// TEXTURING UPLOAD
// ============================================================
function initTexturingUpload() {
  // Mesh upload
  const meshZone = $('#meshUpload');
  const meshInput = $('#meshFileInput');
  meshZone.addEventListener('click', () => meshInput.click());
  meshZone.addEventListener('dragover', (e) => { e.preventDefault(); meshZone.classList.add('dragover'); });
  meshZone.addEventListener('dragleave', () => meshZone.classList.remove('dragover'));
  meshZone.addEventListener('drop', (e) => {
    e.preventDefault();
    meshZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleMeshFile(e.dataTransfer.files[0]);
  });
  meshInput.addEventListener('change', () => {
    if (meshInput.files.length) handleMeshFile(meshInput.files[0]);
  });

  function handleMeshFile(file) {
    state.meshFile = file;
    meshZone.classList.add('has-image');
    $('#meshPlaceholder').hidden = true;
    $('#meshInfo').hidden = false;
    $('#meshFileName').textContent = file.name;
  }

  // Texture reference image upload
  const texZone = $('#texImageUpload');
  const texInput = $('#texImageInput');
  const texPreview = $('#texImagePreview');
  texZone.addEventListener('click', () => texInput.click());
  texZone.addEventListener('dragover', (e) => { e.preventDefault(); texZone.classList.add('dragover'); });
  texZone.addEventListener('dragleave', () => texZone.classList.remove('dragover'));
  texZone.addEventListener('drop', (e) => {
    e.preventDefault();
    texZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleTexImage(e.dataTransfer.files[0]);
  });
  texInput.addEventListener('change', () => {
    if (texInput.files.length) handleTexImage(texInput.files[0]);
  });

  function handleTexImage(file) {
    state.texImage = file;
    texPreview.src = URL.createObjectURL(file);
    texPreview.hidden = false;
    texZone.classList.add('has-image');
  }
}

// ============================================================
// RADIO GROUPS
// ============================================================
function initRadioGroups() {
  $$('.radio-group').forEach((group) => {
    group.querySelectorAll('.radio-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.radio-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  });
}

// ============================================================
// SLIDERS
// ============================================================
function initSliders() {
  const sliderMap = {
    ss_gs: 'ss_gs_val',
    ss_gr: 'ss_gr_val',
    ss_steps: 'ss_steps_val',
    ss_rt: 'ss_rt_val',
    shape_gs: 'shape_gs_val',
    shape_gr: 'shape_gr_val',
    shape_steps: 'shape_steps_val',
    shape_rt: 'shape_rt_val',
    tex_gs: 'tex_gs_val',
    tex_gr: 'tex_gr_val',
    tex_steps: 'tex_steps_val',
    tex_rt: 'tex_rt_val',
  };

  Object.entries(sliderMap).forEach(([sliderId, valId]) => {
    const slider = $(`#${sliderId}`);
    const val = $(`#${valId}`);
    if (slider && val) {
      slider.addEventListener('input', () => {
        val.textContent = slider.value;
      });
    }
  });

  // Decimation slider
  const decSlider = $('#decimationTarget');
  const decVal = $('#dec_val');
  if (decSlider && decVal) {
    decSlider.addEventListener('input', () => {
      const v = parseInt(decSlider.value);
      decVal.textContent = v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : `${Math.round(v / 1000)}K`;
    });
  }

  // Texture size slider
  const texSizeSlider = $('#textureSize');
  const texSizeVal = $('#texsize_val');
  if (texSizeSlider && texSizeVal) {
    texSizeSlider.addEventListener('input', () => {
      texSizeVal.textContent = texSizeSlider.value;
    });
  }
}

// ============================================================
// PRESETS & SURFACE TYPE
// ============================================================
function initPresets() {
  $$('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.preset-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentPreset = btn.dataset.preset;
      applyPreset();
    });
  });
}

function initSurfaceType() {
  $$('.surface-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.surface-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentSurface = btn.dataset.surface;
      applyPreset();
    });
  });
}

function applyPreset() {
  const preset = PRESETS[state.currentPreset];
  const offset = SURFACE_OFFSETS[state.currentSurface];
  if (!preset) return;

  // Helper: set slider value and update displayed label
  function setSlider(id, valId, value, format) {
    const slider = $(`#${id}`);
    const label = $(`#${valId}`);
    if (slider) {
      // Clamp to slider min/max
      const clamped = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), value));
      slider.value = clamped;
      if (label) label.textContent = format ? format(clamped) : clamped;
    }
  }

  // Stage 1: Sparse Structure
  setSlider('ss_gs', 'ss_gs_val', preset.ss_gs + (offset.ss_gs || 0));
  setSlider('ss_gr', 'ss_gr_val', preset.ss_gr);
  setSlider('ss_steps', 'ss_steps_val', preset.ss_steps);
  setSlider('ss_rt', 'ss_rt_val', preset.ss_rt);

  // Stage 2: Shape
  setSlider('shape_gs', 'shape_gs_val', preset.shape_gs + (offset.shape_gs || 0));
  setSlider('shape_gr', 'shape_gr_val', preset.shape_gr);
  setSlider('shape_steps', 'shape_steps_val', preset.shape_steps);
  setSlider('shape_rt', 'shape_rt_val', preset.shape_rt);

  // Stage 3: Texture
  setSlider('tex_gs', 'tex_gs_val', preset.tex_gs + (offset.tex_gs || 0));
  setSlider('tex_gr', 'tex_gr_val', preset.tex_gr);
  setSlider('tex_steps', 'tex_steps_val', preset.tex_steps);
  setSlider('tex_rt', 'tex_rt_val', preset.tex_rt);

  // Export
  setSlider('decimationTarget', 'dec_val', preset.decimation, (v) => {
    return v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : `${Math.round(v / 1000)}K`;
  });
  setSlider('textureSize', 'texsize_val', preset.texSize);
}

// ============================================================
// GENERATE BUTTON
// ============================================================
function initGenerateButton() {
  const btn = $('#generateBtn');
  btn.addEventListener('click', () => {
    if (state.generating) return;
    if (state.activeTab === 'single') generateSingle();
    else if (state.activeTab === 'multiview') generateMultiView();
    else if (state.activeTab === 'texturing') generateTexturing();
  });
}

function getSettings() {
  return {
    resolution: $('.radio-btn.active')?.dataset.value || '1024',
    seed: parseInt($('#seedInput').value) || 0,
    randomize_seed: $('#randomizeSeed').checked,
    ss_guidance_strength: parseFloat($('#ss_gs').value),
    ss_guidance_rescale: parseFloat($('#ss_gr').value),
    ss_sampling_steps: parseInt($('#ss_steps').value),
    ss_rescale_t: parseFloat($('#ss_rt').value),
    shape_guidance_strength: parseFloat($('#shape_gs').value),
    shape_guidance_rescale: parseFloat($('#shape_gr').value),
    shape_sampling_steps: parseInt($('#shape_steps').value),
    shape_rescale_t: parseFloat($('#shape_rt').value),
    tex_guidance_strength: parseFloat($('#tex_gs').value),
    tex_guidance_rescale: parseFloat($('#tex_gr').value),
    tex_sampling_steps: parseInt($('#tex_steps').value),
    tex_rescale_t: parseFloat($('#tex_rt').value),
  };
}

function setGenerating(isGenerating) {
  state.generating = isGenerating;
  const btn = $('#generateBtn');
  btn.disabled = isGenerating;
  btn.classList.toggle('loading', isGenerating);

  const overlay = $('#progressOverlay');
  overlay.hidden = !isGenerating;

  if (!isGenerating) {
    // Reset progress
    updateProgress(0, 'Ready');
  }
}

function updateProgress(percent, stage) {
  const circumference = 2 * Math.PI * 52; // r=52
  const offset = circumference - (percent / 100) * circumference;
  $('#progressRingFill').style.strokeDashoffset = offset;
  $('#progressPercent').textContent = `${Math.round(percent)}%`;
  $('#progressStage').textContent = stage;
}

async function generateSingle() {
  if (!state.singleImage) {
    alert('Please upload an image first.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading image…');

  const settings = getSettings();
  const formData = new FormData();
  formData.append('image', state.singleImage);
  Object.entries(settings).forEach(([k, v]) => formData.append(k, String(v)));

  try {
    updateProgress(10, 'Generating 3D model…');
    const res = await fetch('/api/generate', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }
    const data = await res.json();
    updateProgress(100, 'Done!');
    state.sessionId = data.session_id;
    state.previews = data.previews;
    state.glbUrl = null;
    showPreviews();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

async function generateMultiView() {
  const images = state.multiviewImages.filter((f) => f !== null);
  if (images.length < 2) {
    alert('Please upload at least 2 images for multi-view.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading images…');

  const settings = getSettings();
  const formData = new FormData();
  images.forEach((file, i) => formData.append(`image_${i}`, file));
  Object.entries(settings).forEach(([k, v]) => formData.append(k, String(v)));

  try {
    updateProgress(10, 'Generating 3D model (multi-view)…');
    const res = await fetch('/api/generate-multiview', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }
    const data = await res.json();
    updateProgress(100, 'Done!');
    state.sessionId = data.session_id;
    state.previews = data.previews;
    state.glbUrl = null;
    showPreviews();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

async function generateTexturing() {
  if (!state.meshFile || !state.texImage) {
    alert('Please upload both a mesh file and a reference image.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading files…');

  const settings = getSettings();
  const formData = new FormData();
  formData.append('mesh_file', state.meshFile);
  formData.append('image', state.texImage);
  formData.append('seed', String(settings.seed || 11456));
  formData.append('randomize_seed', 'false');  // Stage 2: fixed seed by default
  formData.append('resolution', settings.resolution);
  formData.append('texture_size', $('#textureSize').value);
  formData.append('tex_guidance_strength', String(settings.tex_guidance_strength));
  formData.append('tex_guidance_rescale', String(settings.tex_guidance_rescale));
  formData.append('tex_sampling_steps', String(settings.tex_sampling_steps));
  formData.append('tex_rescale_t', String(settings.tex_rescale_t));

  try {
    updateProgress(10, 'Re-texturing mesh…');
    const res = await fetch('/api/texturing', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Texturing failed');
    }
    updateProgress(100, 'Done!');
    // Texturing returns GLB directly — show in 3D viewer, download via buttons
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    state.previews = null;
    state.sessionId = null;
    showGlbViewer();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

// ============================================================
// PREVIEW VIEWER
// ============================================================
function showPreviews() {
  if (!state.previews) return;

  $('#emptyState').hidden = true;
  $('#viewerModeBar').hidden = false;
  $('#renderModes').hidden = false;
  $('#angleSliderRow').hidden = false;
  $('#actionBar').hidden = false;

  // Show preview mode
  setViewerMode('preview');
  updatePreviewImage();
}

function updatePreviewImage() {
  if (!state.previews) return;
  const imgs = state.previews[state.currentRenderMode];
  if (!imgs || !imgs[state.currentAngle]) return;
  const img = $('#previewImage');
  img.src = imgs[state.currentAngle];
  img.hidden = false;
  img.classList.add('fade-in');
  setTimeout(() => img.classList.remove('fade-in'), 300);
}

// ============================================================
// VIEWER MODES (Preview / 3D)
// ============================================================
function initViewerModes() {
  $$('.mode-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      setViewerMode(pill.dataset.mode);
    });
  });
}

function setViewerMode(mode) {
  $$('.mode-pill').forEach((p) => p.classList.toggle('active', p.dataset.mode === mode));

  const previewDiv = $('#previewViewer');
  const threeDiv = $('#threeViewer');

  if (mode === 'preview') {
    previewDiv.style.display = 'flex';
    threeDiv.hidden = true;
    $('#renderModes').hidden = false;
    $('#angleSliderRow').hidden = false;
  } else {
    previewDiv.style.display = 'none';
    threeDiv.hidden = false;
    $('#renderModes').hidden = true;
    $('#angleSliderRow').hidden = true;
    if (state.glbUrl) {
      loadGlbInViewer(state.glbUrl);
    } else if (state.sessionId) {
      // Auto-extract GLB for the 3D viewer
      extractGlbForViewer();
    }
  }
}

// Extract GLB silently (no download) and load into 3D viewer
async function extractGlbForViewer() {
  if (state.extracting) return; // prevent double-extraction
  state.extracting = true;

  // Show loading state in the viewer
  showViewerLoading(true);

  const formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('decimation_target', $('#decimationTarget').value);
  formData.append('texture_size', $('#textureSize').value);

  try {
    const res = await fetch('/api/extract-glb', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Extraction failed');
    }
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    showViewerLoading(false);
    loadGlbInViewer(state.glbUrl);
  } catch (err) {
    showViewerLoading(false);
    showViewerError(err.message);
  } finally {
    state.extracting = false;
  }
}

function showViewerLoading(show) {
  let overlay = $('#viewerLoadingOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'viewerLoadingOverlay';
    overlay.className = 'viewer-loading-overlay';
    overlay.innerHTML = `
      <div class="viewer-loading-spinner"></div>
      <p class="viewer-loading-text">Extracting 3D mesh…</p>
    `;
    $('#threeViewer').appendChild(overlay);
  }
  overlay.hidden = !show;
}

function showViewerError(message) {
  let overlay = $('#viewerLoadingOverlay');
  if (overlay) {
    overlay.innerHTML = `<p class="viewer-loading-text" style="color: var(--danger)">⚠ ${message}</p>`;
    overlay.hidden = false;
    setTimeout(() => { overlay.hidden = true; }, 4000);
  }
}

// ============================================================
// RENDER MODES
// ============================================================
function initRenderModes() {
  $$('.render-mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.render-mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentRenderMode = btn.dataset.render;
      updatePreviewImage();
    });
  });
}

// ============================================================
// ANGLE SLIDER
// ============================================================
function initAngleSlider() {
  const slider = $('#angleSlider');
  slider.addEventListener('input', () => {
    state.currentAngle = parseInt(slider.value);
    updatePreviewImage();
  });
}

// ============================================================
// EXTRACT GLB
// ============================================================
function initExtractButton() {
  $('#extractBtn').addEventListener('click', extractGlb);
  $('#extractObjBtn').addEventListener('click', extractObj);
}

async function extractGlb() {
  // If we already have a GLB (e.g. from texturing), download it directly
  if (state.glbUrl && !state.sessionId) {
    triggerDownload(state.glbUrl, 'model.glb');
    return;
  }

  if (!state.sessionId) {
    alert('No model to extract. Generate first.');
    return;
  }

  const btn = $('#extractBtn');
  btn.disabled = true;
  btn.textContent = 'Extracting…';

  const formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('decimation_target', $('#decimationTarget').value);
  formData.append('texture_size', $('#textureSize').value);

  try {
    const res = await fetch('/api/extract-glb', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Extraction failed');
    }
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    triggerDownload(state.glbUrl, 'model.glb');

    // Show 3D view option
    setViewerMode('3d');
    loadGlbInViewer(state.glbUrl);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download GLB
    `;
  }
}

async function extractObj() {
  if (!state.sessionId && !state.glbUrl) {
    alert('No model to extract. Generate first.');
    return;
  }

  const btn = $('#extractObjBtn');
  btn.disabled = true;
  btn.textContent = 'Extracting OBJ…';

  try {
    let blob;

    if (!state.sessionId && state.glbUrl) {
      // Texturing path: convert the GLB blob to OBJ via server
      const glbBlob = await fetch(state.glbUrl).then(r => r.blob());
      const formData = new FormData();
      formData.append('glb_file', glbBlob, 'model.glb');
      const res = await fetch('/api/convert-to-obj', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'OBJ conversion failed');
      }
      blob = await res.blob();
    } else {
      // Generation path: extract OBJ from latents
      const formData = new FormData();
      formData.append('session_id', state.sessionId);
      formData.append('decimation_target', $('#decimationTarget').value);
      formData.append('texture_size', $('#textureSize').value);
      const res = await fetch('/api/extract-obj', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'OBJ extraction failed');
      }
      blob = await res.blob();
    }

    const url = URL.createObjectURL(blob);
    triggerDownload(url, 'model.zip');
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download OBJ
    `;
  }
}

function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ============================================================
// THREE.JS 3D VIEWER
// ============================================================
let threeScene = null;

async function loadGlbInViewer(url) {
  const canvas = $('#threeCanvas');
  const container = $('#threeViewer');

  // Dynamically import three.js
  const THREE = await import('three');
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');

  // Clean up previous scene
  if (threeScene) {
    threeScene.renderer.dispose();
    if (threeScene.resizeObserver) threeScene.resizeObserver.disconnect();
  }

  // Wait one frame for layout reflow (container may have just become visible)
  await new Promise(resolve => requestAnimationFrame(resolve));

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d18);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0.5, 2);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(3, 5, 4);
  scene.add(dirLight);

  const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.5);
  dirLight2.position.set(-3, -2, -4);
  scene.add(dirLight2);

  // Load GLB
  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const model = gltf.scene;
      console.log('[3D Viewer] GLB loaded, meshes:', model.children.length);

      // Center and scale
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      console.log('[3D Viewer] Bounding box size:', size, 'maxDim:', maxDim);

      if (maxDim === 0 || !isFinite(maxDim)) {
        console.warn('[3D Viewer] Model has zero or invalid dimensions');
        return;
      }

      const scale = 1.5 / maxDim;
      model.scale.setScalar(scale);
      model.position.sub(center.multiplyScalar(scale));

      scene.add(model);
      controls.target.set(0, 0, 0);
      controls.update();
    },
    undefined,
    (error) => {
      console.error('[3D Viewer] Failed to load GLB:', error);
    }
  );

  // Animate
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle resize — guard against infinite loops
  let lastW = width, lastH = height;
  const resizeObserver = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0 && (w !== lastW || h !== lastH)) {
      lastW = w;
      lastH = h;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }
  });
  resizeObserver.observe(container);

  threeScene = { scene, camera, renderer, controls, resizeObserver };
}

function showGlbViewer() {
  $('#emptyState').hidden = true;
  $('#viewerModeBar').hidden = false;
  $('#actionBar').hidden = false;
  setViewerMode('3d');
}

// ============================================================
// EXAMPLES
// ============================================================
async function loadExamples() {
  try {
    const res = await fetch('/api/examples');
    if (!res.ok) return;
    const data = await res.json();
    const grid = $('#examplesGrid');
    grid.innerHTML = '';

    data.examples.forEach((name) => {
      const img = document.createElement('img');
      img.src = `/api/examples/${name}`;
      img.alt = name;
      img.className = 'example-thumb';
      img.addEventListener('click', async () => {
        // Fetch the example image and set it as the single image
        const imgRes = await fetch(`/api/examples/${name}`);
        const blob = await imgRes.blob();
        const file = new File([blob], name, { type: blob.type });
        state.singleImage = file;

        const preview = $('#singlePreview');
        preview.src = URL.createObjectURL(file);
        preview.hidden = false;
        $('#singleUpload').classList.add('has-image');

        // Switch to single tab if not there
        if (state.activeTab !== 'single') {
          document.querySelector('.tab-btn[data-tab="single"]').click();
        }
      });
      grid.appendChild(img);
    });
  } catch {
    // Examples not available during dev
  }
}
