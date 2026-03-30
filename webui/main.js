/* ============================================================
   Qwen Building Pipeline — Main Application Logic
   ============================================================ */

// -------- State --------
const state = {
  workspace: null,          // { name, path }
  activeStage: 1,
  generating: false,
  controlImage: null,       // File
  refImage1: null,          // File or { fromStage: url, filename }
  s3RefImage: null,         // File — Stage 03 ref image
  s4RefImage: null,         // File — Stage 04 ref image
  s5SourceImage: null,      // File — Stage 05 source image
  s5OriginalImageData: null,// ImageData — original pixels for live reprocessing
  s5ProcessedCanvas: null,  // Canvas — latest processed result
  selectedGalleryImage: null,
  historyOpen: { 1: false, 2: false, 3: false, 4: false, 5: false },
};

// -------- DOM --------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// -------- Init --------
function initAll() {
  const inits = [
    ['initSettings', initSettings],
    ['initWorkspacePage', initWorkspacePage],
    ['initStageTabs', initStageTabs],
    ['initSliders', initSliders],
    ['initCollapsibles', initCollapsibles],
    ['initUploads', initUploads],
    ['initGenerateButton', initGenerateButton],
    ['initEditButton', initEditButton],
    ['initUseStage1Button', initUseStage1Button],
    ['initPrep3dButton', initPrep3dButton],
    ['initUseStage2Button', initUseStage2Button],
    ['initDelightButton', initDelightButton],
    ['initUseStage3Button', initUseStage3Button],
    ['initHistory', initHistory],
    ['initWorkspaceDialog', initWorkspaceDialog],
    ['initLightbox', initLightbox],
    ['initStage5', initStage5],
  ];
  for (const [name, fn] of inits) {
    try {
      fn();
    } catch (err) {
      console.error(`[INIT] ❌ ${name} CRASHED:`, err);
    }
  }
}

// ES modules are deferred — DOMContentLoaded may have already fired
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAll);
} else {
  initAll();
}

// ============================================================
// WORKSPACE PAGE
// ============================================================
async function initWorkspacePage() {
  await loadSettings();
  await loadWorkspaces();
}

// ============================================================
// SETTINGS (Workspace Root)
// ============================================================
function initSettings() {
  $('#btnChangeRoot').addEventListener('click', () => {
    const dialog = $('#settingsDialog');
    $('#settingsRoot').value = $('#rootPathDisplay').textContent;
    dialog.showModal();
  });

  $('#btnCancelSettings').addEventListener('click', () => {
    $('#settingsDialog').close();
  });

  $('#btnSaveSettings').addEventListener('click', async () => {
    const newRoot = $('#settingsRoot').value.trim();
    if (!newRoot) return;
    try {
      const res = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_root: newRoot }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || 'Failed to save settings');
        return;
      }
      const settings = await res.json();
      $('#rootPathDisplay').textContent = settings.workspace_root;
      $('#settingsDialog').close();
      // Reload workspaces from new root
      await loadWorkspaces();
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  });
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    const settings = await res.json();
    $('#rootPathDisplay').textContent = settings.workspace_root || '~/Pictures/qwen-buildings';
  } catch {
    // ignore
  }
}

async function loadWorkspaces() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch('/api/workspaces', { signal: controller.signal });
    clearTimeout(timeout);
    const data = await res.json();
    renderWorkspaceGrid(data.workspaces || []);
  } catch {
    renderWorkspaceGrid([]);
  }
}

function renderWorkspaceGrid(workspaces) {
  const grid = $('#workspaceGrid');
  grid.innerHTML = '';

  // New workspace card
  const newCard = document.createElement('div');
  newCard.className = 'workspace-card workspace-card-new';
  newCard.innerHTML = `
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    <span class="card-label">New Workspace</span>
  `;
  newCard.addEventListener('click', () => {
    $('#newWorkspaceDialog').showModal();
    $('#newWorkspaceName').value = '';
    $('#newWorkspaceName').focus();
  });
  grid.appendChild(newCard);

  // Existing workspaces
  workspaces.forEach((ws) => {
    const card = document.createElement('div');
    card.className = 'workspace-card';
    const thumbUrl = ws.thumbnail ? `/api/workspaces/${encodeURIComponent(ws.name)}/thumbnail` : '';
    card.innerHTML = `
      ${thumbUrl
        ? `<img class="card-thumb" src="${thumbUrl}" alt="${ws.name}" />`
        : `<div class="card-thumb" style="display:flex;align-items:center;justify-content:center;"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></div>`
      }
      <div class="card-name">${ws.name}</div>
      <div class="card-meta">${ws.image_count || 0} images</div>
    `;
    card.addEventListener('click', () => selectWorkspace(ws));
    grid.appendChild(card);
  });
}

function selectWorkspace(ws) {
  state.workspace = ws;
  $('#workspacePage').style.display = 'none';
  $('#stageLayout').classList.add('active');
  $('#headerWorkspace').style.display = 'flex';
  $('#headerWorkspaceName').textContent = ws.name;
  loadStageGallery(state.activeStage);
  loadHistory(state.activeStage);
}

function goToWorkspacePage() {
  state.workspace = null;
  $('#workspacePage').style.display = '';
  $('#stageLayout').classList.remove('active');
  $('#headerWorkspace').style.display = 'none';
  loadWorkspaces();
}

// ============================================================
// NEW WORKSPACE DIALOG
// ============================================================
function initWorkspaceDialog() {
  $('#btnCancelWorkspace').addEventListener('click', () => {
    $('#newWorkspaceDialog').close();
  });

  $('#btnCreateWorkspace').addEventListener('click', async () => {
    const name = $('#newWorkspaceName').value.trim();
    if (!name) return;
    try {
      const res = await fetch('/api/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || 'Failed to create workspace');
        return;
      }
      const ws = await res.json();
      $('#newWorkspaceDialog').close();
      selectWorkspace(ws);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  });

  $('#newWorkspaceName').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      $('#btnCreateWorkspace').click();
    }
  });

  $('#btnChangeWorkspace').addEventListener('click', goToWorkspacePage);
}

// ============================================================
// STAGE TABS
// ============================================================
function initStageTabs() {
  $$('.stage-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const stage = parseInt(tab.dataset.stage);
      setActiveStage(stage);
    });
  });
}

function setActiveStage(stage) {
  state.activeStage = stage;

  // Update sidebar tabs
  $$('.stage-tab').forEach((t) => {
    t.classList.toggle('active', parseInt(t.dataset.stage) === stage);
  });

  // Update content panels
  $$('.stage-content').forEach((c) => {
    c.classList.toggle('active', parseInt(c.dataset.stage) === stage);
  });

  // Update gallery title
  const titles = { 1: 'Stage 01 Results', 2: 'Stage 02 Results', 3: 'Stage 03 Results', 4: 'Stage 04 Results', 5: 'Stage 05 Results' };
  $('#galleryTitle').textContent = titles[stage] || `Stage ${String(stage).padStart(2, '0')} Results`;

  // Toggle checkerboard class for stage 5 gallery
  $('#galleryGrid').classList.toggle('stage-5-active', stage === 5);

  // Load gallery for this stage
  if (state.workspace) {
    loadStageGallery(stage);
    loadHistory(stage);
  }
}

// ============================================================
// SLIDERS
// ============================================================
function initSliders() {
  const bindings = [
    ['cnScale', 'cnScaleVal'],
    ['cfgScale', 'cfgScaleVal'],
    ['genSteps', 'stepsVal'],
    ['loraScale', 'loraScaleVal'],
    ['editSteps', 'editStepsVal'],
    ['s3LoraScale', 's3LoraScaleVal'],
    ['s3Steps', 's3StepsVal'],
    ['s4LoraScale', 's4LoraScaleVal'],
    ['s4LoraScale2', 's4LoraScale2Val'],
    ['s4Steps', 's4StepsVal'],
    ['s5Thresh', 's5ThreshVal'],
    ['s5Feather', 's5FeatherVal'],
  ];

  bindings.forEach(([sliderId, valId]) => {
    const slider = $(`#${sliderId}`);
    const val = $(`#${valId}`);
    if (slider && val) {
      slider.addEventListener('input', () => {
        val.textContent = slider.value;
      });
    }
  });
}

// ============================================================
// COLLAPSIBLES
// ============================================================
function initCollapsibles() {
  const toggle = $('#negPromptToggle');
  const body = $('#negPromptBody');
  if (toggle && body) {
    toggle.addEventListener('click', () => {
      const open = body.classList.toggle('open');
      toggle.classList.toggle('open', open);
    });
  }
}

// ============================================================
// FILE UPLOADS
// ============================================================
function initUploads() {
  setupUploadZone('controlUpload', 'controlFileInput', 'controlPreview', (file) => {
    state.controlImage = file;
  });

  setupUploadZone('refUpload1', 'refFileInput1', 'refPreview1', (file) => {
    state.refImage1 = file;
  });

  setupUploadZone('s3Upload', 's3FileInput', 's3Preview', (file) => {
    state.s3RefImage = file;
  });

  setupUploadZone('s4Upload', 's4FileInput', 's4Preview', (file) => {
    state.s4RefImage = file;
  });

  setupUploadZone('s5Upload', 's5FileInput', 's5Preview', (file) => {
    state.s5SourceImage = file;
    _s5LoadSourceImage(file);
  });
}

function setupUploadZone(zoneId, inputId, previewId, onFile) {
  const zone = $(`#${zoneId}`);
  const input = $(`#${inputId}`);
  const preview = $(`#${previewId}`);

  zone.addEventListener('click', (e) => {
    if (e.target !== zone && !e.target.closest('.upload-placeholder') && !e.target.closest('.upload-preview')) return;
    input.click();
  });

  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  input.addEventListener('change', () => {
    if (input.files.length) handleFile(input.files[0]);
  });

  function handleFile(file) {
    onFile(file);
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = 'block';
    zone.classList.add('has-image');
  }
}

// ============================================================
// USE STAGE 1 RESULT
// ============================================================
function initUseStage1Button() {
  $('#btnUseStage1').addEventListener('click', async () => {
    if (!state.workspace) return;

    // Load stage 1 images
    try {
      const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/1/images`);
      const data = await res.json();
      const images = data.images || [];

      if (images.length === 0) {
        alert('No Stage 01 images yet. Generate an image first.');
        return;
      }

      // Use the latest image
      const latest = images[images.length - 1];
      const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/1/images/${encodeURIComponent(latest)}`;

      // Download the image and create a File object
      const imgRes = await fetch(imgUrl);
      const blob = await imgRes.blob();
      const file = new File([blob], latest, { type: blob.type });

      state.refImage1 = file;
      const preview = $('#refPreview1');
      preview.src = URL.createObjectURL(blob);
      preview.style.display = 'block';
      $('#refUpload1').classList.add('has-image');
    } catch (err) {
      alert(`Error loading stage 1 images: ${err.message}`);
    }
  });
}

// ============================================================
// GENERATE (Stage 01)
// ============================================================
function initGenerateButton() {
  $('#btnGenerate').addEventListener('click', () => {
    if (state.generating) return;
    generateStage1();
  });
}

async function generateStage1() {
  if (!state.workspace) return;

  state.generating = true;
  setGenerating(true);
  updateProgress(0, 'Uploading…', '', '');

  const formData = new FormData();
  formData.append('workspace', state.workspace.name);
  formData.append('prompt', $('#gen-prompt').value);
  formData.append('negative_prompt', $('#gen-neg-prompt').value);
  formData.append('controlnet_conditioning_scale', $('#cnScale').value);
  formData.append('true_cfg_scale', $('#cfgScale').value);
  formData.append('num_inference_steps', $('#genSteps').value);
  formData.append('aspect_ratio', $('#gen-aspect').value);
  formData.append('sampler_name', $('#gen-sampler').value);
  formData.append('schedule_name', $('#gen-schedule').value);

  const seed = $('#gen-randomize').checked ? -1 : parseInt($('#gen-seed').value);
  formData.append('seed', String(seed));

  if (state.controlImage) {
    formData.append('control_image', state.controlImage);
  }

  try {
    updateProgress(2, 'Sending to server…', '', '');

    // Fire the API request (it will block until generation completes)
    const resPromise = fetch('/api/generate', { method: 'POST', body: formData });

    // Poll for task_id from progress once server starts processing
    // The API response will contain task_id, but we won't get it until it completes.
    // Instead, we start polling the SSE endpoint once we get the response headers.
    // However, since the `/api/generate` blocks, we'll start polling with a small delay.
    const pollPromise = startProgressPolling();

    const res = await resPromise;
    stopProgressPolling();

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }

    const data = await res.json();
    updateProgress(100, 'Done!', '', '');

    // Reload gallery
    await loadStageGallery(1);
    await loadHistory(1);

    if (data.filename) {
      selectGalleryImage(data.filename, 1);
    }
    if (data.seed) {
      $('#gen-seed').value = data.seed;
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    state.generating = false;
    setGenerating(false);
  }
}

// ============================================================
// EDIT (Stage 02)
// ============================================================
function initEditButton() {
  $('#btnEdit').addEventListener('click', () => {
    if (state.generating) return;
    editStage2();
  });
}

async function editStage2() {
  if (!state.workspace) return;
  if (!state.refImage1) {
    alert('Please select a reference image (Picture 1) first.');
    return;
  }

  state.generating = true;
  setGenerating(true);
  updateProgress(0, 'Uploading…', '', '');

  const formData = new FormData();
  formData.append('workspace', state.workspace.name);
  formData.append('prompt', $('#edit-prompt').value);
  formData.append('aspect_ratio', $('#edit-aspect').value);
  formData.append('sampler_name', $('#edit-sampler').value);
  formData.append('schedule_name', $('#edit-schedule').value);
  formData.append('num_inference_steps', $('#editSteps').value);
  formData.append('lora_scale', $('#loraScale').value);
  formData.append('lora_path', $('#edit-lora-path').value);

  const seed = $('#edit-randomize').checked ? -1 : parseInt($('#edit-seed').value);
  formData.append('seed', String(seed));

  formData.append('ref_image_1', state.refImage1);

  try {
    updateProgress(2, 'Sending to server…', '', '');

    const resPromise = fetch('/api/edit', { method: 'POST', body: formData });
    const pollPromise = startProgressPolling();

    const res = await resPromise;
    stopProgressPolling();

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Edit failed');
    }

    const data = await res.json();
    updateProgress(100, 'Done!', '', '');

    await loadStageGallery(2);
    await loadHistory(2);

    if (data.filename) {
      selectGalleryImage(data.filename, 2);
    }
    if (data.seed) {
      $('#edit-seed').value = data.seed;
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    state.generating = false;
    setGenerating(false);
  }
}

// ============================================================
// 3D PREP (Stage 03)
// ============================================================
function initPrep3dButton() {
  $('#btnPrep3d').addEventListener('click', () => {
    if (state.generating) return;
    prepStage3();
  });
}

function initUseStage2Button() {
  $('#btnUseStage2').addEventListener('click', async () => {
    if (!state.workspace) return;
    try {
      const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/2/images`);
      const data = await res.json();
      const images = data.images || [];
      if (images.length === 0) {
        alert('No Stage 02 images yet. Edit an image first.');
        return;
      }
      const latest = images[images.length - 1];
      const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/2/images/${encodeURIComponent(latest)}`;
      const imgRes = await fetch(imgUrl);
      const blob = await imgRes.blob();
      const file = new File([blob], latest, { type: blob.type });
      state.s3RefImage = file;
      const preview = $('#s3Preview');
      preview.src = URL.createObjectURL(blob);
      preview.style.display = 'block';
      $('#s3Upload').classList.add('has-image');
    } catch (err) {
      alert(`Error loading stage 2 images: ${err.message}`);
    }
  });
}

async function prepStage3() {
  if (!state.workspace) return;
  if (!state.s3RefImage) {
    alert('Please select a reference image first.');
    return;
  }

  state.generating = true;
  setGenerating(true);
  updateProgress(0, 'Uploading…', '', '');

  const formData = new FormData();
  formData.append('workspace', state.workspace.name);
  formData.append('prompt', $('#s3-prompt').value);
  formData.append('aspect_ratio', $('#s3-aspect').value);
  formData.append('sampler_name', $('#s3-sampler').value);
  formData.append('schedule_name', $('#s3-schedule').value);
  formData.append('num_inference_steps', $('#s3Steps').value);
  formData.append('lora_scale', $('#s3LoraScale').value);
  formData.append('lora_path', $('#s3-lora-path').value);

  const seed = $('#s3-randomize').checked ? -1 : parseInt($('#s3-seed').value);
  formData.append('seed', String(seed));

  formData.append('ref_image_1', state.s3RefImage);

  try {
    updateProgress(2, 'Sending to server…', '', '');

    const resPromise = fetch('/api/prep3d', { method: 'POST', body: formData });
    const pollPromise = startProgressPolling();

    const res = await resPromise;
    stopProgressPolling();

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '3D Prep failed');
    }

    const data = await res.json();
    updateProgress(100, 'Done!', '', '');

    await loadStageGallery(3);
    await loadHistory(3);

    if (data.filename) {
      selectGalleryImage(data.filename, 3);
    }
    if (data.seed) {
      $('#s3-seed').value = data.seed;
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    state.generating = false;
    setGenerating(false);
  }
}

// ============================================================
// DE-LIGHT (Stage 04)
// ============================================================
function initDelightButton() {
  $('#btnDelight').addEventListener('click', () => {
    if (state.generating) return;
    delightStage4();
  });
}

function initUseStage3Button() {
  $('#btnUseStage3').addEventListener('click', async () => {
    if (!state.workspace) return;
    try {
      const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/3/images`);
      const data = await res.json();
      const images = data.images || [];
      if (images.length === 0) {
        alert('No Stage 03 images yet. Run 3D Prep first.');
        return;
      }
      const latest = images[images.length - 1];
      const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/3/images/${encodeURIComponent(latest)}`;
      const imgRes = await fetch(imgUrl);
      const blob = await imgRes.blob();
      const file = new File([blob], latest, { type: blob.type });
      state.s4RefImage = file;
      const preview = $('#s4Preview');
      preview.src = URL.createObjectURL(blob);
      preview.style.display = 'block';
      $('#s4Upload').classList.add('has-image');
    } catch (err) {
      alert(`Error loading stage 3 images: ${err.message}`);
    }
  });
}

async function delightStage4() {
  if (!state.workspace) return;
  if (!state.s4RefImage) {
    alert('Please select a reference image first.');
    return;
  }

  state.generating = true;
  setGenerating(true);
  updateProgress(0, 'Uploading…', '', '');

  const formData = new FormData();
  formData.append('workspace', state.workspace.name);
  formData.append('prompt', $('#s4-prompt').value);
  formData.append('aspect_ratio', $('#s4-aspect').value);
  formData.append('sampler_name', $('#s4-sampler').value);
  formData.append('schedule_name', $('#s4-schedule').value);
  formData.append('num_inference_steps', $('#s4Steps').value);
  formData.append('lora_scale', $('#s4LoraScale').value);
  formData.append('lora_path', $('#s4-lora-path').value);
  formData.append('lora_scale_2', $('#s4LoraScale2').value);
  formData.append('lora_path_2', $('#s4-lora-path-2').value);

  const seed = $('#s4-randomize').checked ? -1 : parseInt($('#s4-seed').value);
  formData.append('seed', String(seed));

  formData.append('ref_image_1', state.s4RefImage);

  try {
    updateProgress(2, 'Sending to server…', '', '');

    const resPromise = fetch('/api/delight', { method: 'POST', body: formData });
    const pollPromise = startProgressPolling();

    const res = await resPromise;
    stopProgressPolling();

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'De-light failed');
    }

    const data = await res.json();
    updateProgress(100, 'Done!', '', '');

    await loadStageGallery(4);
    await loadHistory(4);

    if (data.filename) {
      selectGalleryImage(data.filename, 4);
    }
    if (data.seed) {
      $('#s4-seed').value = data.seed;
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    state.generating = false;
    setGenerating(false);
  }
}

// ============================================================
// GALLERY
// ============================================================
async function loadStageGallery(stage) {
  if (!state.workspace) return;

  const grid = $('#galleryGrid');

  try {
    const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/${stage}/images`);
    const data = await res.json();
    const images = data.images || [];

    if (images.length === 0) {
      grid.innerHTML = `
        <div class="gallery-empty">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          <p>No images yet. Generate something!</p>
        </div>
      `;
      return;
    }

    grid.innerHTML = '';
    images.forEach((filename) => {
      const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/${stage}/images/${encodeURIComponent(filename)}`;
      const item = document.createElement('div');
      item.className = 'gallery-item fade-in';
      item.innerHTML = `
        <img src="${imgUrl}" alt="${filename}" loading="lazy" />
        <div class="item-actions">
          ${stage === 1 ? `<button class="btn-use-result" data-filename="${filename}" data-target="2" title="Use in Stage 02">→ 02</button>` : ''}
          ${stage === 2 ? `<button class="btn-use-result" data-filename="${filename}" data-target="3" title="Use in Stage 03">→ 03</button>` : ''}
          ${stage === 3 ? `<button class="btn-use-result" data-filename="${filename}" data-target="4" title="Use in Stage 04">→ 04</button>` : ''}
        </div>
      `;

      // Click to view full
      item.addEventListener('click', (e) => {
        if (e.target.closest('.btn-use-result')) return;
        selectGalleryImage(filename, stage);
      });

      // "Use in next stage" shortcut
      const useBtn = item.querySelector('.btn-use-result');
      if (useBtn) {
        useBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const targetStage = parseInt(useBtn.dataset.target);
          const imgRes = await fetch(imgUrl);
          const blob = await imgRes.blob();
          const file = new File([blob], filename, { type: blob.type });

          if (targetStage === 2) {
            state.refImage1 = file;
            const preview = $('#refPreview1');
            preview.src = URL.createObjectURL(blob);
            preview.style.display = 'block';
            $('#refUpload1').classList.add('has-image');
          } else if (targetStage === 3) {
            state.s3RefImage = file;
            const preview = $('#s3Preview');
            preview.src = URL.createObjectURL(blob);
            preview.style.display = 'block';
            $('#s3Upload').classList.add('has-image');
          } else if (targetStage === 4) {
            state.s4RefImage = file;
            const preview = $('#s4Preview');
            preview.src = URL.createObjectURL(blob);
            preview.style.display = 'block';
            $('#s4Upload').classList.add('has-image');
          }
          setActiveStage(targetStage);
        });
      }

      grid.appendChild(item);
    });
  } catch (err) {
    console.error('Gallery load error:', err);
  }
}

function selectGalleryImage(filename, stage) {
  state.selectedGalleryImage = { filename, stage };
  $$('.gallery-item').forEach((item) => item.classList.remove('selected'));
  $$('.gallery-item img').forEach((img) => {
    if (img.alt === filename) {
      img.closest('.gallery-item').classList.add('selected');
    }
  });

  // Open lightbox
  const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/${stage}/images/${encodeURIComponent(filename)}`;
  openLightbox(imgUrl, filename, stage);
}

// ============================================================
// PROGRESS (SSE-based)
// ============================================================
let _progressPollId = null;
let _progressStartTime = null;
let _sseStepTimes = []; // track per-step times for ETA

function setGenerating(isGenerating) {
  const overlay = $('#progressOverlay');
  overlay.style.display = isGenerating ? 'flex' : 'none';

  const btns = ['#btnGenerate', '#btnEdit', '#btnPrep3d', '#btnDelight'];
  btns.forEach((sel) => {
    const btn = $(sel);
    if (btn) {
      btn.disabled = isGenerating;
      btn.classList.toggle('loading', isGenerating);
    }
  });

  if (isGenerating) {
    _progressStartTime = Date.now();
    _sseStepTimes = [];
  }
}

function updateProgress(percent, stageText, stepText, timeText) {
  const fill = $('#progressFill');
  fill.style.width = `${percent}%`;
  $('#progressText').textContent = stageText || '';
  $('#progressStep').textContent = [stepText, timeText].filter(Boolean).join('  ·  ');
}

function formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

function startProgressPolling() {
  // Poll the SSE progress endpoint every 500ms
  // The task_id isn't known yet (it's in the response), so we poll a "latest" endpoint
  // Instead, we'll poll the general progress store via a simpler approach:
  // We poll /api/progress/latest which returns the most recent active task
  let lastStep = -1;

  _progressPollId = setInterval(async () => {
    try {
      const res = await fetch('/api/progress/latest');
      if (!res.ok) return;
      const info = await res.json();

      if (!info || info.done) return;

      const { step, total, stage, message, vram, vram_gb } = info;

      // Stage text
      const stageText = stage || 'Processing…';

      // Step counter
      let stepText = '';
      if (total > 0) {
        stepText = `Step ${step}/${total}`;
      }

      // Time tracking
      if (step > 0 && step !== lastStep) {
        _sseStepTimes.push(Date.now());
        lastStep = step;
      }

      // ETA calculation
      let timeText = '';
      const elapsed = Date.now() - _progressStartTime;
      timeText = `Elapsed: ${formatDuration(elapsed)}`;

      if (step > 1 && total > 0 && _sseStepTimes.length >= 2) {
        const recent = _sseStepTimes.slice(-10);
        let avgStepMs = 0;
        for (let i = 1; i < recent.length; i++) {
          avgStepMs += recent[i] - recent[i - 1];
        }
        avgStepMs /= (recent.length - 1);
        const remaining = (total - step) * avgStepMs;
        timeText += `  ·  ETA: ${formatDuration(remaining)}`;
      }

      // VRAM info — multi-GPU
      if (vram && typeof vram === 'object') {
        const parts = Object.entries(vram).map(([gpu, v]) =>
          `${gpu.toUpperCase()}: ${v.alloc}/${v.total}GB`
        );
        timeText += `  ·  ${parts.join('  ')}`;
      } else if (vram_gb > 0) {
        timeText += `  ·  VRAM: ${vram_gb} GB`;
      }

      // Progress bar percentage
      const percent = total > 0 ? Math.max(2, Math.round((step / total) * 100)) : 2;

      updateProgress(percent, stageText, stepText, timeText);
    } catch {
      // ignore polling errors
    }
  }, 500);
}

function stopProgressPolling() {
  if (_progressPollId) {
    clearInterval(_progressPollId);
    _progressPollId = null;
  }
}

// ============================================================
// HISTORY
// ============================================================
function initHistory() {
  ['1', '2', '3', '4', '5'].forEach((stage) => {
    const toggle = $(`#historyToggle${stage}`);
    const list = $(`#historyList${stage}`);
    if (toggle && list) {
      toggle.addEventListener('click', () => {
        const open = list.classList.toggle('open');
        state.historyOpen[stage] = open;
      });
    }
  });
}

async function loadHistory(stage) {
  if (!state.workspace) return;
  const list = $(`#historyList${stage}`);
  if (!list) return;

  try {
    const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/history?stage=${stage}`);
    const data = await res.json();
    const entries = data.history || [];

    list.innerHTML = '';

    if (entries.length === 0) {
      list.innerHTML = '<div class="history-item" style="justify-content:center;color:var(--cloud-dim);cursor:default;font-size:12px;">No history yet</div>';
      return;
    }

    entries.reverse().forEach((entry) => {
      const item = document.createElement('div');
      item.className = 'history-item';

      const thumbUrl = entry.thumbnail
        ? `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/${stage}/images/${encodeURIComponent(entry.thumbnail)}`
        : '';

      // Control image URL (stage 1 only, dot-file served via the image endpoint)
      const ctrlUrl = (stage == 1 && entry.control_image)
        ? `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/1/images/${encodeURIComponent(entry.control_image)}`
        : '';

      item.innerHTML = `
        ${thumbUrl ? `<img class="hist-thumb" src="${thumbUrl}" alt="" />` : '<div class="hist-thumb"></div>'}
        <div class="hist-info">
          <div class="hist-prompt">${entry.prompt || '(no prompt)'}</div>
          <div class="hist-meta">seed:${entry.seed || '?'} • ${entry.timestamp || ''}${ctrlUrl ? ' • <span class="hist-ctrl-badge" title="Click to load ControlNet image">🎛</span>' : ''}</div>
        </div>
      `;

      // Click row → load params
      item.addEventListener('click', (e) => {
        if (e.target.closest('.hist-ctrl-badge')) return; // handled below
        loadHistoryEntry(entry, stage);
      });

      // Click control badge → load control image into upload zone
      const ctrlBadge = item.querySelector('.hist-ctrl-badge');
      if (ctrlBadge && ctrlUrl) {
        ctrlBadge.style.cursor = 'pointer';
        ctrlBadge.addEventListener('click', async (e) => {
          e.stopPropagation();
          try {
            const imgRes = await fetch(ctrlUrl);
            const blob = await imgRes.blob();
            const file = new File([blob], entry.control_image, { type: blob.type });
            state.controlImage = file;
            const preview = $('#controlPreview');
            preview.src = URL.createObjectURL(blob);
            preview.style.display = 'block';
            $('#controlUpload').classList.add('has-image');
            setActiveStage(1);
          } catch (err) {
            console.error('Failed to load control image:', err);
          }
        });
      }

      list.appendChild(item);
    });
  } catch (err) {
    console.error('History load error:', err);
  }
}

function loadHistoryEntry(entry, stage) {
  if (stage === 1 || stage === '1') {
    if (entry.prompt) $('#gen-prompt').value = entry.prompt;
    if (entry.negative_prompt) $('#gen-neg-prompt').value = entry.negative_prompt;
    if (entry.controlnet_conditioning_scale != null) {
      $('#cnScale').value = entry.controlnet_conditioning_scale;
      $('#cnScaleVal').textContent = entry.controlnet_conditioning_scale;
    }
    if (entry.true_cfg_scale != null) {
      $('#cfgScale').value = entry.true_cfg_scale;
      $('#cfgScaleVal').textContent = entry.true_cfg_scale;
    }
    if (entry.num_inference_steps != null) {
      $('#genSteps').value = entry.num_inference_steps;
      $('#stepsVal').textContent = entry.num_inference_steps;
    }
    if (entry.aspect_ratio) $('#gen-aspect').value = entry.aspect_ratio;
    if (entry.sampler_name) $('#gen-sampler').value = entry.sampler_name;
    if (entry.schedule_name) $('#gen-schedule').value = entry.schedule_name;
    if (entry.seed != null) {
      $('#gen-seed').value = entry.seed;
      $('#gen-randomize').checked = false;
    }
    // Auto-load control image if available
    if (entry.control_image && state.workspace) {
      const ctrlUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/1/images/${encodeURIComponent(entry.control_image)}`;
      fetch(ctrlUrl).then(r => r.blob()).then(blob => {
        const file = new File([blob], entry.control_image, { type: blob.type });
        state.controlImage = file;
        const preview = $('#controlPreview');
        preview.src = URL.createObjectURL(blob);
        preview.style.display = 'block';
        $('#controlUpload').classList.add('has-image');
      }).catch(() => {});
    }
  } else if (stage === 2 || stage === '2') {
    if (entry.prompt) $('#edit-prompt').value = entry.prompt;
    if (entry.num_inference_steps != null) {
      $('#editSteps').value = entry.num_inference_steps;
      $('#editStepsVal').textContent = entry.num_inference_steps;
    }
    if (entry.lora_scale != null) {
      $('#loraScale').value = entry.lora_scale;
      $('#loraScaleVal').textContent = entry.lora_scale;
    }
    if (entry.lora_path) $('#edit-lora-path').value = entry.lora_path;
    if (entry.aspect_ratio) $('#edit-aspect').value = entry.aspect_ratio;
    if (entry.sampler_name) $('#edit-sampler').value = entry.sampler_name;
    if (entry.schedule_name) $('#edit-schedule').value = entry.schedule_name;
    if (entry.seed != null) {
      $('#edit-seed').value = entry.seed;
      $('#edit-randomize').checked = false;
    }
  } else if (stage === 3 || stage === '3') {
    if (entry.prompt) $('#s3-prompt').value = entry.prompt;
    if (entry.num_inference_steps != null) {
      $('#s3Steps').value = entry.num_inference_steps;
      $('#s3StepsVal').textContent = entry.num_inference_steps;
    }
    if (entry.lora_scale != null) {
      $('#s3LoraScale').value = entry.lora_scale;
      $('#s3LoraScaleVal').textContent = entry.lora_scale;
    }
    if (entry.lora_path) $('#s3-lora-path').value = entry.lora_path;
    if (entry.aspect_ratio) $('#s3-aspect').value = entry.aspect_ratio;
    if (entry.sampler_name) $('#s3-sampler').value = entry.sampler_name;
    if (entry.schedule_name) $('#s3-schedule').value = entry.schedule_name;
    if (entry.seed != null) {
      $('#s3-seed').value = entry.seed;
      $('#s3-randomize').checked = false;
    }
  } else if (stage === 4 || stage === '4') {
    if (entry.prompt) $('#s4-prompt').value = entry.prompt;
    if (entry.num_inference_steps != null) {
      $('#s4Steps').value = entry.num_inference_steps;
      $('#s4StepsVal').textContent = entry.num_inference_steps;
    }
    if (entry.lora_scale != null) {
      $('#s4LoraScale').value = entry.lora_scale;
      $('#s4LoraScaleVal').textContent = entry.lora_scale;
    }
    if (entry.lora_path) $('#s4-lora-path').value = entry.lora_path;
    if (entry.lora_scale_2 != null) {
      $('#s4LoraScale2').value = entry.lora_scale_2;
      $('#s4LoraScale2Val').textContent = entry.lora_scale_2;
    }
    if (entry.lora_path_2) $('#s4-lora-path-2').value = entry.lora_path_2;
    if (entry.aspect_ratio) $('#s4-aspect').value = entry.aspect_ratio;
    if (entry.sampler_name) $('#s4-sampler').value = entry.sampler_name;
    if (entry.schedule_name) $('#s4-schedule').value = entry.schedule_name;
    if (entry.seed != null) {
      $('#s4-seed').value = entry.seed;
      $('#s4-randomize').checked = false;
    }
  }
}

// ============================================================
// LIGHTBOX / COMPARE
// ============================================================
let _lightboxCompareMode = false;
let _lightboxRefUrl = null;

function initLightbox() {
  // Close button
  $('#lightboxClose').addEventListener('click', closeLightbox);

  // Backdrop click
  $('#lightboxBackdrop').addEventListener('click', closeLightbox);

  // Esc key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && $('#lightbox').classList.contains('open')) {
      closeLightbox();
    }
  });

  // Compare toggle
  $('#lightboxCompareToggle').addEventListener('click', () => {
    _lightboxCompareMode = !_lightboxCompareMode;
    $('#lightboxCompareToggle').classList.toggle('active', _lightboxCompareMode);
    $('#lightboxSingle').style.display = _lightboxCompareMode ? 'none' : 'flex';
    $('#lightboxCompare').style.display = _lightboxCompareMode ? 'flex' : 'none';

    if (_lightboxCompareMode) {
      // Set up compare images
      const resultSrc = $('#lightboxImage').src;
      $('#compareAfter').src = resultSrc;
      if (_lightboxRefUrl) {
        $('#compareBefore').src = _lightboxRefUrl;
      }
      // Reset divider to 50%
      setComparePosition(50);
    }
  });

  // Compare drag
  initCompareDrag();
}

function openLightbox(imageUrl, filename, stage) {
  const lb = $('#lightbox');
  const img = $('#lightboxImage');

  img.src = imageUrl;
  $('#lightboxFilename').textContent = filename;

  // Reset compare mode
  _lightboxCompareMode = false;
  $('#lightboxCompareToggle').classList.remove('active');
  $('#lightboxSingle').style.display = 'flex';
  $('#lightboxCompare').style.display = 'none';

  // Show compare button for stages that have a reference image
  const compareBtn = $('#lightboxCompareToggle');
  const refImageForStage = {
    2: state.refImage1,
    3: state.s3RefImage,
    4: state.s4RefImage,
    5: state.s5SourceImage,
  };
  const refImage = refImageForStage[stage];
  if (refImage) {
    _lightboxRefUrl = URL.createObjectURL(refImage);
    compareBtn.style.display = 'flex';
  } else {
    compareBtn.style.display = 'none';
    _lightboxRefUrl = null;
  }

  lb.classList.add('open');
}

function closeLightbox() {
  const lb = $('#lightbox');
  lb.classList.remove('open');
  // Clean up blob URLs
  if (_lightboxRefUrl) {
    URL.revokeObjectURL(_lightboxRefUrl);
    _lightboxRefUrl = null;
  }
}

function setComparePosition(pct) {
  const clip = $('#compareClip');
  const divider = $('#compareDivider');
  clip.style.width = `${pct}%`;
  divider.style.left = `${pct}%`;
  // Adjust before image width to maintain proper scaling
  const before = $('#compareBefore');
  if (pct > 0) {
    before.style.width = `${(100 / pct) * 100}%`;
  }
}

function initCompareDrag() {
  const container = document.getElementById('compareContainer');
  const divider = document.getElementById('compareDivider');
  let dragging = false;

  function updateFromEvent(e) {
    const rect = container.getBoundingClientRect();
    const x = (e.clientX || e.touches?.[0]?.clientX || 0) - rect.left;
    const pct = Math.max(2, Math.min(98, (x / rect.width) * 100));
    setComparePosition(pct);
  }

  divider.addEventListener('pointerdown', (e) => {
    dragging = true;
    divider.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  divider.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    updateFromEvent(e);
  });

  divider.addEventListener('pointerup', () => {
    dragging = false;
  });

  // Also allow clicking anywhere on the container to move divider
  container.addEventListener('click', (e) => {
    if (e.target.closest('.compare-divider')) return;
    updateFromEvent(e);
  });
}

// ============================================================
// STAGE 05: REMOVE WHITE BACKGROUND
// ============================================================
let _s5RafId = null;  // for debouncing live preview

function initStage5() {
  // "Use Stage 04 Result" button
  $('#btnUseStage4').addEventListener('click', async () => {
    if (!state.workspace) return;
    try {
      const res = await fetch(`/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/4/images`);
      const data = await res.json();
      const images = data.images || [];
      if (images.length === 0) {
        alert('No Stage 04 images yet. Run De-light first.');
        return;
      }
      const latest = images[images.length - 1];
      const imgUrl = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/4/images/${encodeURIComponent(latest)}`;
      const imgRes = await fetch(imgUrl);
      const blob = await imgRes.blob();
      const file = new File([blob], latest, { type: blob.type });
      state.s5SourceImage = file;
      const preview = $('#s5Preview');
      preview.src = URL.createObjectURL(blob);
      preview.style.display = 'block';
      $('#s5Upload').classList.add('has-image');
      _s5LoadSourceImage(file);
    } catch (err) {
      alert(`Error loading stage 4 images: ${err.message}`);
    }
  });

  // Live preview: update when sliders change
  const threshSlider = $('#s5Thresh');
  const featherSlider = $('#s5Feather');

  const debouncedUpdate = () => {
    if (!$('#s5LivePreview').checked) return;
    if (_s5RafId) cancelAnimationFrame(_s5RafId);
    _s5RafId = requestAnimationFrame(() => {
      _s5ProcessAndPreview();
      _s5RafId = null;
    });
  };

  threshSlider.addEventListener('input', debouncedUpdate);
  featherSlider.addEventListener('input', debouncedUpdate);

  // Remove Background button
  $('#btnRemoveBg').addEventListener('click', () => {
    if (!state.s5OriginalImageData) {
      alert('Please load a source image first.');
      return;
    }
    _s5ProcessAndPreview();
  });

  // Export PNG button
  $('#btnExportPng').addEventListener('click', async () => {
    if (!state.s5ProcessedCanvas) {
      alert('Process an image first (click "Remove Background").');
      return;
    }
    if (!state.workspace) return;

    try {
      // Convert canvas to blob
      const blob = await new Promise((resolve) => {
        state.s5ProcessedCanvas.toBlob(resolve, 'image/png');
      });

      const formData = new FormData();
      formData.append('workspace', state.workspace.name);
      formData.append('image', blob, `bg_removed_${Date.now()}.png`);
      formData.append('threshold', $('#s5Thresh').value);
      formData.append('feather', $('#s5Feather').value);

      const res = await fetch('/api/export-alpha', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Export failed');
      }

      const data = await res.json();

      // Reload gallery
      await loadStageGallery(5);

      // Also trigger a browser download
      const a = document.createElement('a');
      a.href = `/api/workspaces/${encodeURIComponent(state.workspace.name)}/stages/5/images/${encodeURIComponent(data.filename)}`;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert(`Export error: ${err.message}`);
    }
  });
}

/**
 * Load a source image File into pixel data for canvas processing.
 */
function _s5LoadSourceImage(file) {
  const img = new Image();
  img.onload = () => {
    // Create offscreen canvas with original dimensions
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    state.s5OriginalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Trigger initial preview if live preview is on
    if ($('#s5LivePreview').checked) {
      _s5ProcessAndPreview();
    }
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}

/**
 * Process the source image with threshold + feather and show in gallery.
 */
function _s5ProcessAndPreview() {
  const originalData = state.s5OriginalImageData;
  if (!originalData) return;

  const threshold = parseInt($('#s5Thresh').value);
  const feather = parseInt($('#s5Feather').value);
  const w = originalData.width;
  const h = originalData.height;

  // Create working canvas
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  // Copy original data
  const newData = ctx.createImageData(w, h);
  const src = originalData.data;
  const dst = newData.data;

  // Step 1: Compute alpha based on threshold
  // For feathering, we first compute a "whiteness distance" then apply feather gradient
  const alphaMap = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const r = src[idx];
    const g = src[idx + 1];
    const b = src[idx + 2];

    // All channels must be >= threshold to be considered white
    const minChannel = Math.min(r, g, b);

    if (minChannel >= threshold) {
      alphaMap[i] = 0; // fully transparent
    } else if (feather > 0 && minChannel >= threshold - feather * 8) {
      // Feather zone: gradual transparency
      const dist = threshold - minChannel;
      const featherRange = feather * 8;
      alphaMap[i] = Math.min(1.0, dist / featherRange);
    } else {
      alphaMap[i] = 1.0; // fully opaque
    }
  }

  // Step 2: Optional spatial feather (simple box blur on alpha)
  let finalAlpha = alphaMap;
  if (feather > 0) {
    finalAlpha = _s5BlurAlpha(alphaMap, w, h, feather);
  }

  // Step 3: Write pixels
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    dst[idx] = src[idx];
    dst[idx + 1] = src[idx + 1];
    dst[idx + 2] = src[idx + 2];
    dst[idx + 3] = Math.round(finalAlpha[i] * 255);
  }

  ctx.putImageData(newData, 0, 0);
  state.s5ProcessedCanvas = canvas;

  // Update the gallery area with the preview
  _s5ShowPreview(canvas, threshold, feather);
}

/**
 * Simple box blur on the alpha channel for edge feathering.
 */
function _s5BlurAlpha(alpha, w, h, radius) {
  const result = new Float32Array(w * h);
  const r = Math.max(1, radius);

  // Horizontal pass
  const temp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      let count = 0;
      for (let dx = -r; dx <= r; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < w) {
          sum += alpha[y * w + nx];
          count++;
        }
      }
      temp[y * w + x] = sum / count;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      let count = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny >= 0 && ny < h) {
          sum += temp[ny * w + x];
          count++;
        }
      }
      result[y * w + x] = sum / count;
    }
  }

  return result;
}

/**
 * Show the processed canvas in the gallery area with checkerboard background.
 */
function _s5ShowPreview(canvas, threshold, feather) {
  const grid = $('#galleryGrid');

  // Count transparent pixels for info
  const ctx = canvas.getContext('2d');
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let transparent = 0;
  let semiTransparent = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] === 0) transparent++;
    else if (data[i] < 255) semiTransparent++;
  }
  const totalPixels = canvas.width * canvas.height;
  const pctRemoved = ((transparent / totalPixels) * 100).toFixed(1);

  grid.innerHTML = `
    <div class="s5-live-preview" style="grid-column: 1 / -1;">
      <div class="s5-preview-canvas-wrap">
        <canvas id="s5ResultCanvas" width="${canvas.width}" height="${canvas.height}"></canvas>
      </div>
      <div class="s5-preview-info">
        ${canvas.width}×${canvas.height} · threshold: ${threshold} · feather: ${feather}px
        · ${pctRemoved}% removed · ${semiTransparent} semi-transparent pixels
      </div>
    </div>
  `;

  // Draw the processed canvas
  const displayCanvas = document.getElementById('s5ResultCanvas');
  const displayCtx = displayCanvas.getContext('2d');
  displayCtx.drawImage(canvas, 0, 0);
}
