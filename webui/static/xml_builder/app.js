const baseSelect = document.getElementById("baseSelect");
const replaceToggle = document.getElementById("replaceToggle");
const entitySelect = document.getElementById("entitySelect");
const densitySlider = document.getElementById("densitySlider");
const snippetList = document.getElementById("snippetList");
const previewTitle = document.getElementById("previewTitle");
const previewMeta = document.getElementById("previewMeta");
const previewXml = document.getElementById("previewXml");
const insertBtn = document.getElementById("insertBtn");
const outputXml = document.getElementById("outputXml");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");
const validateBtn = document.getElementById("validateBtn");
const runBtn = document.getElementById("runBtn");
const messagesEl = document.getElementById("messages");
const clearMessagesBtn = document.getElementById("clearMessagesBtn");
const configForm = document.getElementById("configForm");
const toolbox = document.getElementById("toolbox");
const toolboxOverlay = document.getElementById("toolboxOverlay");
const toolboxOverlayBack = document.getElementById("toolboxOverlayBack");
const toolboxOverlayTitle = document.getElementById("toolboxOverlayTitle");
const toolboxOverlayList = document.getElementById("toolboxOverlayList");
const workflowStrip = document.getElementById("workflowStrip");
const workflowItems = document.getElementById("workflowItems");
const paletteDialog = document.getElementById("paletteDialog");
const paletteSearch = document.getElementById("paletteSearch");
const paletteList = document.getElementById("paletteList");
const paletteHint = document.getElementById("paletteHint");
const plotFile = document.getElementById("plotFile");
const plotX = document.getElementById("plotX");
const plotY = document.getElementById("plotY");
const plotCanvas = document.getElementById("plotCanvas");
const plotMeta = document.getElementById("plotMeta");
const runStatus = document.getElementById("runStatus");
const runLog = document.getElementById("runLog");
const openDashboardBtn = document.getElementById("openDashboardBtn");
const runOutputsHint = document.getElementById("runOutputsHint");
const runsList = document.getElementById("runsList");
const refreshRunsBtn = document.getElementById("refreshRunsBtn");
const runsFilter = document.getElementById("runsFilter");
const runNameInput = document.getElementById("runNameInput");
const runsDatalist = document.getElementById("runsDatalist");
const openRunBtn = document.getElementById("openRunBtn");
const runPathInput = document.getElementById("runPathInput");
const openRunPathBtn = document.getElementById("openRunPathBtn");
const browseRunPathBtn = document.getElementById("browseRunPathBtn");
const runPathHint = document.getElementById("runPathHint");
const runPathDialog = document.getElementById("runPathDialog");
const closeRunPathDialog = document.getElementById("closeRunPathDialog");
const browseUpBtn = document.getElementById("browseUpBtn");
const browsePathInput = document.getElementById("browsePathInput");
const browseGoBtn = document.getElementById("browseGoBtn");
const browseList = document.getElementById("browseList");
const browseUseBtn = document.getElementById("browseUseBtn");
const browseOpenBtn = document.getElementById("browseOpenBtn");

let catalog = null;
let snippets = [];
let selectedSnippetId = null;
let editableSnippetElement = null;
let messages = [];
let plotRows = [];
let plotColumns = [];
let plotTimer = null;
let entityOptionCache = new Map();
let paletteMode = "snippets";
let paletteContextEntity = null;
let paletteReplaceRange = null;
let paletteActiveOptions = [];
let workflow = [];
let activeToolboxGroupId = null;
let activeRunJobId = null;
let runPollTimer = null;
let cachedRuns = [];
let browseCurrentPath = "";

const dynamicToolRegistry = new Map();

function makeInstanceId() {
  if (typeof crypto !== "undefined" && crypto && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `wf_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

function workflowMarkerText(instanceId, toolId) {
  return `prlo-workflow id=${instanceId} tool=${toolId}`;
}

function parseWorkflowMarker(text) {
  const raw = safeText(text).trim();
  const match = raw.match(/^prlo-workflow\s+id=([^\s]+)\s+tool=([^\s]+)\s*$/);
  if (!match) {
    return null;
  }
  return { instanceId: match[1], toolId: match[2] };
}

function normalizeWorkflowEntries(raw) {
  if (!Array.isArray(raw)) {
    return [];
  }
  const normalized = [];
  for (const entry of raw) {
    if (typeof entry === "string") {
      normalized.push({ instanceId: null, toolId: entry });
      continue;
    }
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const toolId = typeof entry.toolId === "string" ? entry.toolId : null;
    if (!toolId) {
      continue;
    }
    const instanceId = typeof entry.instanceId === "string" ? entry.instanceId : null;
    normalized.push({ instanceId, toolId });
  }
  return normalized;
}

const toolboxItems = [
  {
    id: "testinfo",
    label: "TestInfo",
    icon: "/static/xml_builder/icons/testinfo.svg",
    kind: "section",
    section: "TestInfo",
  },
  {
    id: "runinfo",
    label: "RunInfo",
    icon: "/static/xml_builder/icons/runinfo.svg",
    kind: "section",
    section: "RunInfo",
  },
  {
    id: "variablegroups",
    label: "VariableGroups",
    icon: "/static/xml_builder/icons/variablegroups.svg",
    kind: "section",
    section: "VariableGroups",
  },
  {
    id: "dataobjects",
    label: "DataObjects",
    icon: "/static/xml_builder/icons/dataobjects.svg",
    kind: "group",
    entity: "DataObjects",
    dynamic: true,
    children: [
      {
        id: "dataobjects_section",
        label: "DataObjects (block)",
        icon: "/static/xml_builder/icons/dataobjects.svg",
        kind: "section",
        section: "DataObjects",
      },
      {
        id: "pointset",
        label: "PointSet",
        icon: "/static/xml_builder/icons/pointset.svg",
        kind: "child",
        section: "DataObjects",
        xml: '<PointSet name="">\n</PointSet>',
      },
      {
        id: "historyset",
        label: "HistorySet",
        icon: "/static/xml_builder/icons/historyset.svg",
        kind: "child",
        section: "DataObjects",
        xml: '<HistorySet name="">\n</HistorySet>',
      },
      {
        id: "dataset",
        label: "DataSet",
        icon: "/static/xml_builder/icons/dataset.svg",
        kind: "child",
        section: "DataObjects",
        xml: '<DataSet name="">\n</DataSet>',
      },
    ],
  },
  {
    id: "databases",
    label: "Databases",
    icon: "/static/xml_builder/icons/databases.svg",
    kind: "group",
    entity: "Databases",
    dynamic: true,
    children: [
      {
        id: "databases_section",
        label: "Databases (block)",
        icon: "/static/xml_builder/icons/databases.svg",
        kind: "section",
        section: "Databases",
      },
      {
        id: "hdf5",
        label: "HDF5",
        icon: "/static/xml_builder/icons/hdf5.svg",
        kind: "child",
        section: "Databases",
        xml: '<HDF5 name="">\n</HDF5>',
      },
      {
        id: "netcdf",
        label: "NetCDF",
        icon: "/static/xml_builder/icons/netcdf.svg",
        kind: "child",
        section: "Databases",
        xml: '<NetCDF name="">\n</NetCDF>',
      },
    ],
  },
  {
    id: "files",
    label: "Files",
    icon: "/static/xml_builder/icons/files.svg",
    kind: "group",
    entity: "Files",
    dynamic: true,
    children: [
      {
        id: "files_section",
        label: "Files (block)",
        icon: "/static/xml_builder/icons/files.svg",
        kind: "section",
        section: "Files",
      },
    ],
  },
  {
    id: "functions",
    label: "Functions",
    icon: "/static/xml_builder/icons/functions.svg",
    kind: "group",
    entity: "Functions",
    dynamic: true,
    children: [
      {
        id: "functions_section",
        label: "Functions (block)",
        icon: "/static/xml_builder/icons/functions.svg",
        kind: "section",
        section: "Functions",
      },
    ],
  },
  {
    id: "distributions",
    label: "Distributions",
    icon: "/static/xml_builder/icons/distributions.svg",
    kind: "group",
    entity: "Distributions",
    dynamic: true,
    children: [
      {
        id: "distributions_section",
        label: "Distributions (block)",
        icon: "/static/xml_builder/icons/distributions.svg",
        kind: "section",
        section: "Distributions",
      },
    ],
  },
  {
    id: "samplers",
    label: "Samplers",
    icon: "/static/xml_builder/icons/samplers.svg",
    kind: "group",
    entity: "Samplers",
    dynamic: true,
    children: [
      {
        id: "samplers_section",
        label: "Samplers (block)",
        icon: "/static/xml_builder/icons/samplers.svg",
        kind: "section",
        section: "Samplers",
      },
    ],
  },
  {
    id: "optimizers",
    label: "Optimizers",
    icon: "/static/xml_builder/icons/optimizers.svg",
    kind: "group",
    entity: "Optimizers",
    dynamic: true,
    children: [
      {
        id: "optimizers_section",
        label: "Optimizers (block)",
        icon: "/static/xml_builder/icons/optimizers.svg",
        kind: "section",
        section: "Optimizers",
      },
    ],
  },
  {
    id: "models",
    label: "Models",
    icon: "/static/xml_builder/icons/models.svg",
    kind: "group",
    entity: "Models",
    dynamic: true,
    children: [
      {
        id: "models_section",
        label: "Models (block)",
        icon: "/static/xml_builder/icons/models.svg",
        kind: "section",
        section: "Models",
      },
      {
        id: "model_code",
        label: "Code (interface)",
        icon: "/static/xml_builder/icons/models_code.svg",
        kind: "child",
        section: "Models",
        xml: '<Code name="">\n</Code>',
      },
      {
        id: "model_external",
        label: "ExternalModel (Python)",
        icon: "/static/xml_builder/icons/models_externalmodel.svg",
        kind: "child",
        section: "Models",
        xml: '<ExternalModel name="">\n</ExternalModel>',
      },
      {
        id: "model_rom",
        label: "ROM (surrogate)",
        icon: "/static/xml_builder/icons/models_rom.svg",
        kind: "child",
        section: "Models",
        xml: '<ROM name="">\n</ROM>',
      },
      {
        id: "model_ensemble",
        label: "EnsembleModel",
        icon: "/static/xml_builder/icons/models_ensemblemodel.svg",
        kind: "child",
        section: "Models",
        xml: '<EnsembleModel name="">\n</EnsembleModel>',
      },
      {
        id: "model_postprocessor",
        label: "PostProcessor (Model)",
        icon: "/static/xml_builder/icons/models_postprocessor.svg",
        kind: "child",
        section: "Models",
        xml: '<PostProcessor name="">\n</PostProcessor>',
      },
    ],
  },
  {
    id: "postprocessors",
    label: "PostProcessors",
    icon: "/static/xml_builder/icons/postprocessors.svg",
    kind: "group",
    entity: "PostProcessors",
    dynamic: true,
    children: [
      {
        id: "postprocessors_section",
        label: "PostProcessors (block)",
        icon: "/static/xml_builder/icons/postprocessors.svg",
        kind: "section",
        section: "PostProcessors",
      },
    ],
  },
  {
    id: "metrics",
    label: "Metrics",
    icon: "/static/xml_builder/icons/metrics.svg",
    kind: "group",
    entity: "Metrics",
    dynamic: true,
    children: [
      {
        id: "metrics_section",
        label: "Metrics (block)",
        icon: "/static/xml_builder/icons/metrics.svg",
        kind: "section",
        section: "Metrics",
      },
    ],
  },
  {
    id: "outstreams",
    label: "OutStreams",
    icon: "/static/xml_builder/icons/outstreams.svg",
    kind: "group",
    entity: "OutStreams",
    dynamic: true,
    children: [
      {
        id: "outstreams_section",
        label: "OutStreams (block)",
        icon: "/static/xml_builder/icons/outstreams.svg",
        kind: "section",
        section: "OutStreams",
      },
    ],
  },
  {
    id: "steps",
    label: "Steps",
    icon: "/static/xml_builder/icons/steps.svg",
    kind: "group",
    entity: "Steps",
    dynamic: true,
    children: [
      {
        id: "steps_section",
        label: "Steps (block)",
        icon: "/static/xml_builder/icons/steps.svg",
        kind: "section",
        section: "Steps",
      },
    ],
  },
];

function flattenToolboxItems(items) {
  const flat = [];
  for (const item of items) {
    flat.push(item);
    if (item.kind === "group" && Array.isArray(item.children)) {
      flat.push(...item.children);
    }
  }
  return flat;
}

function toolboxItemById(itemId) {
  if (dynamicToolRegistry.has(itemId)) {
    return dynamicToolRegistry.get(itemId);
  }
  const dynMatch = safeText(itemId).match(/^dyn:([^:]+):(.+)$/);
  if (dynMatch) {
    const entity = dynMatch[1];
    const tag = dynMatch[2];
    const parent = toolboxItems.find((item) => item && item.kind === "group" && item.entity === entity) || null;
    const icon =
      (subtypeIconOverrides[entity] && subtypeIconOverrides[entity][tag]) ||
      (parent && parent.icon) ||
      "/static/xml_builder/icons/dataobjects.svg";
    const tool = {
      id: itemId,
      label: tag,
      icon,
      kind: "child",
      section: entity,
      xml: `<${tag} name="">\n</${tag}>`,
    };
    registerDynamicToolItem(tool);
    return tool;
  }
  return flattenToolboxItems(toolboxItems).find((item) => item.id === itemId) || null;
}

function registerDynamicToolItem(item) {
  if (!item || !item.id) {
    return;
  }
  dynamicToolRegistry.set(item.id, item);
}

function dynamicToolId(entity, tag) {
  return `dyn:${entity}:${tag}`;
}

const subtypeIconOverrides = {
  DataObjects: {
    PointSet: "/static/xml_builder/icons/pointset.svg",
    HistorySet: "/static/xml_builder/icons/historyset.svg",
    DataSet: "/static/xml_builder/icons/dataset.svg",
  },
  Databases: {
    HDF5: "/static/xml_builder/icons/hdf5.svg",
    NetCDF: "/static/xml_builder/icons/netcdf.svg",
  },
  Models: {
    EnsembleModel: "/static/xml_builder/icons/models_ensemblemodel.svg",
  },
  Optimizers: {
    GradientDescent: "/static/xml_builder/icons/optimizers_gradient_descent.svg",
    SimulatedAnnealing: "/static/xml_builder/icons/optimizers_simulated_annealing.svg",
    GeneticAlgorithm: "/static/xml_builder/icons/optimizers_genetic_algorithm.svg",
  },
  Steps: {
    SingleRun: "/static/xml_builder/icons/steps_singlerun.svg",
    MultiRun: "/static/xml_builder/icons/steps_multirun.svg",
    IOStep: "/static/xml_builder/icons/steps_iostep.svg",
    PostProcess: "/static/xml_builder/icons/steps_postprocess.svg",
    RomTrainer: "/static/xml_builder/icons/steps_romtrainer.svg",
  },
};

function persistWorkflow() {
  window.localStorage.setItem("prlo.workflow", JSON.stringify(workflow));
}

function loadWorkflow() {
  const stored = window.localStorage.getItem("prlo.workflow");
  if (!stored) {
    return [];
  }
  try {
    const parsed = JSON.parse(stored);
    return normalizeWorkflowEntries(parsed);
  } catch (_err) {
    return [];
  }
}

function addToWorkflow(entry) {
  if (!entry || typeof entry !== "object") {
    return;
  }
  if (!entry.toolId) {
    return;
  }
  workflow.push({ instanceId: entry.instanceId || null, toolId: entry.toolId });
  persistWorkflow();
  renderWorkflow();
}

function removeWorkflowByInstanceId(instanceId) {
  if (!instanceId) {
    return;
  }
  const idx = workflow.findIndex((entry) => entry && entry.instanceId === instanceId);
  if (idx < 0) {
    return;
  }
  workflow.splice(idx, 1);
  persistWorkflow();
  renderWorkflow();
}

function renderWorkflow() {
  if (!workflowItems) {
    return;
  }
  workflowItems.innerHTML = "";
  workflow.forEach((entry, idx) => {
    const item = toolboxItemById(entry.toolId);
    if (!item) {
      return;
    }
    const chip = document.createElement("div");
    chip.className = "chip";
    if (entry.instanceId) {
      chip.dataset.instanceId = entry.instanceId;
    }
    chip.dataset.toolId = entry.toolId;
    const img = document.createElement("img");
    img.alt = item.label;
    img.src = item.icon;
    const label = document.createElement("div");
    label.className = "chip__label";
    label.textContent = item.label;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip__remove";
    remove.title = "Remove from workflow + output XML";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      // Prefer stable identity removal in case the workflow re-syncs between render and click.
      const instanceId = entry.instanceId || null;
      if (instanceId) {
        removeWorkflowEntry({ instanceId, toolId: entry.toolId });
        return;
      }
      removeFromWorkflow(idx);
    });
    chip.appendChild(img);
    chip.appendChild(label);
    chip.appendChild(remove);
    workflowItems.appendChild(chip);
  });
}

const ravenEntities = [
  // Based on `doc/user_manual/raven_user_manual.pdf` (root is <Simulation>), plus test harness metadata.
  "TestInfo",
  "RunInfo",
  "Files",
  "VariableGroups",
  "Distributions",
  "Samplers",
  "Optimizers",
  "DataObjects",
  "Databases",
  "OutStreams",
  "Models",
  "PostProcessors",
  "Functions",
  "Metrics",
  "Steps",
];

const densityPresets = [
  {
    gap: "18px",
    padding: "16px 14px",
    title: "15px",
    meta: "12px",
  },
  {
    gap: "12px",
    padding: "13px 12px",
    title: "14px",
    meta: "12px",
  },
  {
    gap: "8px",
    padding: "10px 10px 9px 10px",
    title: "13px",
    meta: "11px",
  },
  {
    gap: "6px",
    padding: "8px 9px",
    title: "12px",
    meta: "11px",
  },
  {
    gap: "4px",
    padding: "6px 8px",
    title: "12px",
    meta: "10px",
  },
];

function skeletonXml() {
  return `<?xml version="1.0" ?>\n<Simulation>\n</Simulation>\n`;
}

function safeText(value) {
  return value == null ? "" : String(value);
}

function normalizeWhitespace(text) {
  return safeText(text).replace(/\s+/g, " ").trim();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatXml(xmlString) {
  const xml = xmlString.replace(/>\s+</g, "><").trim();
  const parts = xml.split(/(?=<)|(?<=>)/g).filter((part) => part !== "");
  const lines = [];
  let indent = 0;
  for (const part of parts) {
    if (part.startsWith("</")) {
      indent = Math.max(0, indent - 1);
    }
    if (part.startsWith("<") && part.endsWith(">")) {
      lines.push(`${"  ".repeat(indent)}${part}`);
      if (part.startsWith("<") && !part.startsWith("</") && !part.endsWith("/>") && !part.startsWith("<?") && !part.startsWith("<!")) {
        indent += 1;
      }
      continue;
    }
    const text = part.trim();
    if (text) {
      lines.push(`${"  ".repeat(indent)}${text}`);
    }
  }
  return lines.join("\n") + "\n";
}

function parseXml(xmlText) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlText, "application/xml");
  const errors = doc.getElementsByTagName("parsererror");
  if (errors.length > 0) {
    const message = errors[0].textContent || "Invalid XML";
    throw new Error(message);
  }
  return doc;
}

function stripXmlDeclaration(xmlText) {
  return safeText(xmlText).replace(/^\s*<\?xml[^>]*\?>\s*/i, "");
}

function nowStamp() {
  const date = new Date();
  return date.toLocaleTimeString();
}

function logMessage(type, text) {
  messages.unshift({ time: nowStamp(), type, text: safeText(text) });
  if (messages.length > 250) {
    messages = messages.slice(0, 250);
  }
  renderMessages();
}

function setRunStatus(text) {
  if (!runStatus) {
    return;
  }
  runStatus.textContent = safeText(text);
}

function setRunLog(text) {
  if (!runLog) {
    return;
  }
  runLog.textContent = safeText(text);
  runLog.scrollTop = runLog.scrollHeight;
}

function setDashboardLink(href) {
  if (!openDashboardBtn) {
    return;
  }
  if (!href) {
    openDashboardBtn.hidden = true;
    openDashboardBtn.href = "#";
    return;
  }
  openDashboardBtn.hidden = false;
  openDashboardBtn.href = href;
}

function setRunOutputsHint(text) {
  if (!runOutputsHint) {
    return;
  }
  runOutputsHint.textContent = safeText(text);
}

function renderRunsList(runs) {
  if (!runsList) {
    return;
  }
  runsList.innerHTML = "";
  if (!runs || runs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "hint";
    empty.textContent = "No existing runs found.";
    runsList.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const row = document.createElement("div");
    row.className = "runlist__item";
    const name = document.createElement("div");
    name.className = "runlist__name";
    name.textContent = run.display_name || run.name || "run";
    const link = document.createElement("a");
    link.className = "runlist__link";
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "Open dashboard";
    link.href = `/api/xml-builder/run-folder/${encodeURIComponent(run.name)}/dashboard`;
    row.appendChild(name);
    row.appendChild(link);
    runsList.appendChild(row);
  });
}

function updateRunsDatalist(runs) {
  if (!runsDatalist) {
    return;
  }
  runsDatalist.innerHTML = "";
  runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = run.display_name || run.name;
    runsDatalist.appendChild(option);
  });
}

function applyRunsFilter() {
  if (!runsFilter) {
    renderRunsList(cachedRuns);
    return;
  }
  const query = runsFilter.value.trim().toLowerCase();
  if (!query) {
    renderRunsList(cachedRuns);
    return;
  }
  const filtered = cachedRuns.filter((run) => {
    const name = (run.name || "").toLowerCase();
    const display = (run.display_name || "").toLowerCase();
    return name.includes(query) || display.includes(query);
  });
  renderRunsList(filtered);
}

async function loadRuns() {
  if (!runsList) {
    return;
  }
  runsList.innerHTML = "<div class=\"hint\">Loading runs...</div>";
  try {
    const response = await fetch("/api/xml-builder/runs");
    if (!response.ok) {
      throw new Error(`Runs request failed: ${response.status}`);
    }
    const payload = await response.json();
    cachedRuns = Array.isArray(payload.runs) ? payload.runs : [];
    updateRunsDatalist(cachedRuns);
    applyRunsFilter();
  } catch (error) {
    cachedRuns = [];
    renderRunsList([]);
    logMessage("error", error.message);
  }
}

function setRunPathHint(text) {
  if (!runPathHint) {
    return;
  }
  runPathHint.textContent = safeText(text);
}

function openRunPathDashboard(path) {
  const trimmed = (path || "").trim();
  if (!trimmed) {
    logMessage("warn", "Enter an output folder path.");
    return;
  }
  const url = `/api/xml-builder/run-path/dashboard?path=${encodeURIComponent(trimmed)}`;
  window.open(url, "_blank", "noreferrer");
}

function renderBrowseList(dirs) {
  if (!browseList) {
    return;
  }
  browseList.innerHTML = "";
  if (!dirs || dirs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "hint";
    empty.textContent = "No subfolders found.";
    browseList.appendChild(empty);
    return;
  }
  dirs.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "browse__item";
    const name = document.createElement("div");
    name.className = "browse__name";
    name.textContent = entry.name;
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.textContent = "Open";
    openBtn.addEventListener("click", () => {
      loadBrowseDirs(entry.path);
    });
    row.appendChild(name);
    row.appendChild(openBtn);
    browseList.appendChild(row);
  });
}

async function loadBrowseDirs(path) {
  if (!browseList || !browsePathInput) {
    return;
  }
  browseList.innerHTML = "<div class=\"hint\">Loading folders...</div>";
  try {
    const params = path ? `?path=${encodeURIComponent(path)}` : "";
    const response = await fetch(`/api/xml-builder/list-dirs${params}`);
    if (!response.ok) {
      throw new Error(`Directory request failed: ${response.status}`);
    }
    const payload = await response.json();
    browseCurrentPath = payload.path || "";
    browsePathInput.value = browseCurrentPath;
    if (browseUpBtn) {
      browseUpBtn.disabled = !payload.parent;
      browseUpBtn.dataset.target = payload.parent || "";
    }
    renderBrowseList(Array.isArray(payload.dirs) ? payload.dirs : []);
  } catch (error) {
    renderBrowseList([]);
    logMessage("error", error.message);
  }
}

function openBrowseDialog() {
  if (!runPathDialog || typeof runPathDialog.showModal !== "function") {
    logMessage("warn", "Folder browser not available.");
    return;
  }
  const startPath = runPathInput && runPathInput.value.trim() ? runPathInput.value.trim() : "";
  runPathDialog.showModal();
  loadBrowseDirs(startPath);
}

function closeBrowseDialog() {
  if (runPathDialog && typeof runPathDialog.close === "function") {
    runPathDialog.close();
  }
}

async function startRun() {
  const xml = outputXml.value || "";
  if (!xml.trim()) {
    logMessage("warn", "Output XML is empty.");
    return;
  }
  const context_path = baseSelect && baseSelect.value ? baseSelect.value : "";
  setRunStatus("Submitting run…");
  setRunLog("");
  setDashboardLink("");
  setRunOutputsHint("");
  const response = await fetch("/api/xml-builder/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ xml, context_path }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Run request failed: ${response.status}`);
  }
  const payload = await response.json();
  activeRunJobId = payload.job_id;
  logMessage("ok", `Run started: ${activeRunJobId}`);
  if (payload.workdir) {
    logMessage("info", `Run directory: ${payload.workdir}`);
  }
  setDashboardLink(`/api/xml-builder/run/${encodeURIComponent(activeRunJobId)}/dashboard`);
  pollRunStatus();
}

async function pollRunStatus() {
  if (!activeRunJobId) {
    return;
  }
  if (runPollTimer) {
    clearTimeout(runPollTimer);
    runPollTimer = null;
  }
  let payload = null;
  try {
    const params = new URLSearchParams({ tail_lines: "400" });
    const response = await fetch(`/api/xml-builder/run/${encodeURIComponent(activeRunJobId)}?${params.toString()}`);
    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || `Run status failed: ${response.status}`);
    }
    payload = await response.json();
  } catch (error) {
    setRunStatus(`Run error: ${error.message}`);
    logMessage("error", error.message);
    return;
  }

  const status = payload.status || "unknown";
  const rc = payload.returncode;
  const header = rc == null ? `Run ${activeRunJobId}: ${status}` : `Run ${activeRunJobId}: ${status} (rc=${rc})`;
  setRunStatus(header);
  setRunLog(payload.tail || "");
  if (payload.raven_workdir) {
    setRunOutputsHint(`Outputs: ${payload.raven_workdir}`);
  } else if (payload.job_dir) {
    setRunOutputsHint(`Job dir: ${payload.job_dir}`);
  }
  if (payload.dashboard_url) {
    setDashboardLink(payload.dashboard_url);
  } else {
    setDashboardLink(`/api/xml-builder/run/${encodeURIComponent(activeRunJobId)}/dashboard`);
  }
  if (status === "running" || status === "queued") {
    runPollTimer = setTimeout(pollRunStatus, 1200);
    return;
  }
  if (status === "done") {
    logMessage("ok", `Run completed: ${activeRunJobId}`);
    return;
  }
  logMessage("warn", `Run finished with status=${status}`);
}

function renderMessages() {
  messagesEl.innerHTML = "";
  if (messages.length === 0) {
    messagesEl.textContent = "No messages yet.";
    return;
  }
  for (const message of messages) {
    const row = document.createElement("div");
    row.className = "messages__item";
    const time = document.createElement("span");
    time.className = "messages__time";
    time.textContent = message.time;
    const type = document.createElement("span");
    type.className = "messages__type";
    type.textContent = message.type;
    const text = document.createElement("span");
    text.textContent = message.text;
    row.appendChild(time);
    row.appendChild(type);
    row.appendChild(text);
    messagesEl.appendChild(row);
  }
}

function serializeXml(doc) {
  const serializer = new XMLSerializer();
  const xmlText = serializer.serializeToString(doc);
  return formatXml(xmlText);
}

function serializeElement(element) {
  const serializer = new XMLSerializer();
  return formatXml(serializer.serializeToString(element));
}

function insertText(textarea, text, replaceRange = null) {
  const start = replaceRange ? replaceRange.start : textarea.selectionStart;
  const end = replaceRange ? replaceRange.end : textarea.selectionEnd;
  const before = textarea.value.slice(0, start);
  const after = textarea.value.slice(end);
  textarea.value = before + text + after;
  const cursor = start + text.length;
  textarea.selectionStart = cursor;
  textarea.selectionEnd = cursor;
  textarea.focus();
}

function currentLineIndent(text, index) {
  const lineStart = text.lastIndexOf("\n", index - 1) + 1;
  const match = text.slice(lineStart, index).match(/^[ \t]*/);
  return match ? match[0] : "";
}

function indentBlock(block, baseIndent) {
  const lines = block.split("\n");
  return lines
    .map((line, idx) => (idx === 0 ? line : baseIndent + line))
    .join("\n");
}

function buildSnippetTitle(snippet) {
  const label = normalizeWhitespace(snippet.label);
  const name = normalizeWhitespace(snippet.name);
  if (name) {
    return `${snippet.section} / ${snippet.tag} : ${name}`;
  }
  if (label) {
    return `${snippet.section} / ${snippet.tag} : ${label}`;
  }
  return `${snippet.section} / ${snippet.tag}`;
}

function buildSnippetMeta(snippet) {
  const pieces = [];
  if (snippet.source) {
    pieces.push(snippet.source);
  }
  return pieces.join(" · ");
}

function buildOptionTitle(option) {
  return safeText(option.tag);
}

function buildOptionMeta(option) {
  return normalizeWhitespace(option.description || "");
}

function createSnippetItem(snippet, onActivate) {
  const div = document.createElement("div");
  div.className = "list__item";
  div.dataset.snippetId = snippet.id;
  if (snippet.id === selectedSnippetId) {
    div.classList.add("list__item--active");
  }
  const title = document.createElement("div");
  title.className = "list__title";
  title.textContent = buildSnippetTitle(snippet);
  const meta = document.createElement("div");
  meta.className = "list__meta";
  meta.textContent = buildSnippetMeta(snippet);
  div.appendChild(title);
  div.appendChild(meta);
  div.addEventListener("click", () => onActivate(snippet, false));
  div.addEventListener("dblclick", () => onActivate(snippet, true));
  return div;
}

function createOptionItem(option, onActivate) {
  const div = document.createElement("div");
  div.className = "list__item";
  const title = document.createElement("div");
  title.className = "list__title";
  title.textContent = buildOptionTitle(option);
  const meta = document.createElement("div");
  meta.className = "list__meta";
  meta.textContent = buildOptionMeta(option);
  div.appendChild(title);
  if (meta.textContent) {
    div.appendChild(meta);
  }
  div.addEventListener("click", () => onActivate(option, false));
  div.addEventListener("dblclick", () => onActivate(option, true));
  return div;
}

function renderSnippetList(items, options = {}) {
  const groupBySection = options.groupBySection !== false;
  snippetList.innerHTML = "";

  const onActivate = (activatedSnippet, shouldInsert) => {
    selectSnippet(activatedSnippet.id);
    if (shouldInsert) {
      try {
        insertSnippetIntoOutput(activatedSnippet);
        logMessage("insert", `Inserted ${buildSnippetTitle(activatedSnippet)}`);
      } catch (error) {
        logMessage("error", error.message);
        alert(error.message);
      }
    }
  };

  if (!groupBySection) {
    for (const snippet of items) {
      snippetList.appendChild(createSnippetItem(snippet, onActivate));
    }
    return;
  }

  const grouped = new Map();
  for (const snippet of items) {
    const section = safeText(snippet.section) || "Other";
    if (!grouped.has(section)) {
      grouped.set(section, []);
    }
    grouped.get(section).push(snippet);
  }
  const sortedSections = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
  for (const section of sortedSections) {
    const details = document.createElement("details");
    details.className = "group";
    details.open = section === "RunInfo" || section === "Steps";
    const summary = document.createElement("summary");
    summary.textContent = section;
    details.appendChild(summary);
    const itemsEl = document.createElement("div");
    itemsEl.className = "group__items";
    for (const snippet of grouped.get(section)) {
      itemsEl.appendChild(createSnippetItem(snippet, onActivate));
    }
    details.appendChild(itemsEl);
    snippetList.appendChild(details);
  }
}

function selectSnippet(snippetId) {
  selectedSnippetId = snippetId;
  const snippet = snippets.find((item) => item.id === snippetId);
  if (!snippet) {
    insertBtn.disabled = true;
    previewTitle.textContent = "Select a block";
    previewMeta.textContent = "";
    previewXml.value = "";
    editableSnippetElement = null;
    configForm.innerHTML = "";
    render();
    return;
  }
  previewTitle.textContent = buildSnippetTitle(snippet);
  previewMeta.textContent = buildSnippetMeta(snippet);
  previewXml.value = snippet.xml;
  try {
    editableSnippetElement = parseSnippetElement(stripXmlDeclaration(snippet.xml));
    renderConfigForm(editableSnippetElement);
    previewXml.value = serializeElement(editableSnippetElement);
  } catch (error) {
    editableSnippetElement = null;
    configForm.innerHTML = "";
    logMessage("warn", `Could not parse snippet for editing: ${error.message}`);
  }
  insertBtn.disabled = false;
  render();
}

function currentSnippet() {
  return snippets.find((item) => item.id === selectedSnippetId) || null;
}

function ensureSection(outputDoc, sectionTag) {
  const root = outputDoc.documentElement;
  let section = root.querySelector(`:scope > ${CSS.escape(sectionTag)}`);
  if (!section) {
    section = outputDoc.createElement(sectionTag);
    root.appendChild(section);
  }
  return section;
}

function insertToolboxItem(item) {
  const outputDoc = parseXml(outputXml.value);
  if (outputDoc.documentElement.tagName !== "Simulation") {
    throw new Error(`Root tag must be <Simulation>, got <${outputDoc.documentElement.tagName}>`);
  }
  const root = outputDoc.documentElement;
  const sectionNode = ensureSection(outputDoc, item.section);

  if (item.kind === "section") {
    // Track the section with a comment marker so workflow strip can remove it later.
    const instanceId = makeInstanceId();
    root.insertBefore(outputDoc.createComment(workflowMarkerText(instanceId, item.id)), sectionNode);
    outputXml.value = serializeXml(outputDoc);
    scheduleValidation();
    return { instanceId, toolId: item.id };
  }

  const snippetXml = stripXmlDeclaration(item.xml || "");
  const blockNode = parseSnippetElement(snippetXml);
  const replaceMode = replaceToggle.checked;
  const existing = findMatching(sectionNode, blockNode, replaceMode);
  if (existing) {
    const prev = existing.previousSibling;
    if (prev && prev.nodeType === Node.COMMENT_NODE) {
      const marker = parseWorkflowMarker(prev.data || "");
      if (marker && marker.instanceId) {
        sectionNode.removeChild(prev);
        removeWorkflowByInstanceId(marker.instanceId);
      }
    }
    sectionNode.removeChild(existing);
  }
  const instanceId = makeInstanceId();
  const markerNode = outputDoc.createComment(workflowMarkerText(instanceId, item.id));
  const imported = outputDoc.importNode(blockNode, true);
  sectionNode.appendChild(markerNode);
  sectionNode.appendChild(imported);
  outputXml.value = serializeXml(outputDoc);
  scheduleValidation();
  return { instanceId, toolId: item.id };
}

function findMatching(sectionNode, blockNode, replaceMode) {
  if (!replaceMode) {
    return null;
  }
  const tagName = blockNode.tagName;
  const name = blockNode.getAttribute("name");
  const candidates = Array.from(sectionNode.children).filter((child) => child.tagName === tagName);
  if (name) {
    return candidates.find((child) => child.getAttribute("name") === name) || null;
  }
  if (candidates.length === 1) {
    return candidates[0];
  }
  return null;
}

function parseSnippetElement(snippetXml) {
  const wrapperDoc = parseXml(`<Wrapper>${snippetXml}</Wrapper>`);
  const blockNode = wrapperDoc.documentElement.firstElementChild;
  if (!blockNode) {
    throw new Error("Snippet XML is empty.");
  }
  return blockNode;
}

function insertSnippetIntoOutput(snippet) {
  const outputDoc = parseXml(outputXml.value);
  const snippetXml = stripXmlDeclaration(previewXml.value || snippet.xml);
  const blockNode = parseSnippetElement(snippetXml);
  const sectionNode = ensureSection(outputDoc, snippet.section);
  const replaceMode = replaceToggle.checked;
  const existing = findMatching(sectionNode, blockNode, replaceMode);
  if (existing) {
    sectionNode.removeChild(existing);
  }
  const imported = outputDoc.importNode(blockNode, true);
  sectionNode.appendChild(imported);
  outputXml.value = serializeXml(outputDoc);
  scheduleValidation();
}

function previewToolboxItem(item) {
  selectedSnippetId = null;
  insertBtn.disabled = true;
  editableSnippetElement = null;
  configForm.innerHTML = "";
  const header = item.kind === "section" ? item.section : `${item.section} / ${parseSnippetElement(stripXmlDeclaration(item.xml || "<Unknown/>")).tagName}`;
  previewTitle.textContent = `Toolbox / ${item.label}`;
  previewMeta.textContent = header;
  if (item.kind === "section") {
    previewXml.value = `<${item.section}>\n</${item.section}>`;
  } else {
    previewXml.value = item.xml || "";
  }
  render();
}

function insertToolboxItemAndTrack(item) {
  const entry = insertToolboxItem(item);
  if (entry) {
    addToWorkflow(entry);
  }
}

function removeToolboxBlockFromOutput(entry) {
  const tool = toolboxItemById(entry.toolId);
  if (!tool) {
    return false;
  }
  const outputDoc = parseXml(outputXml.value);
  if (outputDoc.documentElement.tagName !== "Simulation") {
    throw new Error(`Root tag must be <Simulation>, got <${outputDoc.documentElement.tagName}>`);
  }
  const root = outputDoc.documentElement;

  // Preferred removal path: marker comment with instance id.
  if (entry.instanceId) {
    const walker = outputDoc.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
    let node = walker.nextNode();
    while (node) {
      const marker = parseWorkflowMarker(node.data || "");
      if (marker && marker.instanceId === entry.instanceId) {
        const parent = node.parentNode;
        if (!parent) {
          break;
        }
        let cursor = node.nextSibling;
        while (cursor && cursor.nodeType !== Node.ELEMENT_NODE) {
          cursor = cursor.nextSibling;
        }
        if (cursor && cursor.nodeType === Node.ELEMENT_NODE) {
          parent.removeChild(cursor);
        }
        parent.removeChild(node);
        outputXml.value = serializeXml(outputDoc);
        scheduleValidation();
        return true;
      }
      node = walker.nextNode();
    }
  }

  // Fallback: remove the first matching block by tag within the expected section.
  if (tool.kind === "section") {
    const sectionNode = root.querySelector(`:scope > ${CSS.escape(tool.section)}`);
    if (sectionNode) {
      root.removeChild(sectionNode);
      outputXml.value = serializeXml(outputDoc);
      scheduleValidation();
      return true;
    }
    return false;
  }

  const sectionNode = root.querySelector(`:scope > ${CSS.escape(tool.section)}`);
  if (!sectionNode) {
    return false;
  }
  const tagName = parseSnippetElement(stripXmlDeclaration(tool.xml || "<Unknown/>")).tagName;
  const candidate = Array.from(sectionNode.children).find((child) => child.tagName === tagName);
  if (candidate) {
    sectionNode.removeChild(candidate);
    outputXml.value = serializeXml(outputDoc);
    scheduleValidation();
    return true;
  }
  return false;
}

function removeWorkflowEntry(entry) {
  if (!entry) {
    return;
  }
  const instanceId = entry.instanceId || null;
  const toolId = entry.toolId || null;
  const removedFromXml = removeToolboxBlockFromOutput(entry);

  if (instanceId) {
    workflow = workflow.filter((candidate) => candidate && candidate.instanceId !== instanceId);
  } else if (toolId) {
    const idx = workflow.findIndex((candidate) => candidate && candidate.toolId === toolId);
    if (idx >= 0) {
      workflow.splice(idx, 1);
    }
  }

  persistWorkflow();
  renderWorkflow();

  if (removedFromXml) {
    logMessage("ok", `Removed ${toolId || "tool"} from workflow/output`);
  } else {
    logMessage(
      "warn",
      `Removed ${toolId || "tool"} from workflow, but could not locate its XML block (is the XML still valid?)`
    );
  }
}

function removeFromWorkflow(index) {
  const entry = workflow[index];
  if (!entry) {
    return;
  }
  try {
    removeWorkflowEntry(entry);
  } catch (error) {
    logMessage("error", error.message);
  }
}

function syncWorkflowFromOutput() {
  try {
    const outputDoc = parseXml(outputXml.value);
    const root = outputDoc.documentElement;
    if (!root || root.tagName !== "Simulation") {
      return;
    }
    const found = [];
    const walker = outputDoc.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
    let node = walker.nextNode();
    while (node) {
      const marker = parseWorkflowMarker(node.data || "");
      if (marker && marker.instanceId && marker.toolId) {
        found.push({ instanceId: marker.instanceId, toolId: marker.toolId });
      }
      node = walker.nextNode();
    }
    // If any markers exist, treat them as authoritative for the workflow strip.
    if (found.length > 0) {
      workflow = found;
      persistWorkflow();
      renderWorkflow();
    }
  } catch (_err) {
    // ignore invalid XML while typing
  }
}

function render() {
  const selectedEntity = entitySelect ? entitySelect.value : "__all__";
  if (!selectedEntity || selectedEntity === "__all__") {
    renderSnippetList(snippets, { groupBySection: true });
    return;
  }
  const filtered = snippets.filter((snippet) => snippet.section === selectedEntity);
  renderSnippetList(filtered, { groupBySection: false });
  if (filtered.length === 0) {
    logMessage("warn", `No example blocks found for ${selectedEntity} in plugins/PRLO/examples.`);
  }
}

function populateEntitySelect() {
  if (!entitySelect) {
    return;
  }
  entitySelect.innerHTML = "";
  const all = document.createElement("option");
  all.value = "__all__";
  all.textContent = "All entities";
  entitySelect.appendChild(all);

  for (const entity of ravenEntities) {
    const option = document.createElement("option");
    option.value = entity;
    option.textContent = entity;
    entitySelect.appendChild(option);
  }

  const stored = window.localStorage.getItem("prlo.entityFilter") || "__all__";
  const selectedOption = Array.from(entitySelect.options).find(
    (opt) => opt.value === stored
  );
  entitySelect.value = selectedOption ? stored : "__all__";
}

function renderConfigForm(element) {
  configForm.innerHTML = "";
  const tagRow = document.createElement("div");
  tagRow.className = "config__row";
  tagRow.innerHTML = `<label>Tag</label><div>${safeText(element.tagName)}</div>`;
  configForm.appendChild(tagRow);

  const nameAttr = element.getAttribute("name");
  const nameRow = document.createElement("div");
  nameRow.className = "config__row";
  const nameInput = document.createElement("input");
  nameInput.value = nameAttr || "";
  nameInput.placeholder = "(optional)";
  nameInput.addEventListener("input", () => {
    if (nameInput.value.trim() === "") {
      element.removeAttribute("name");
    } else {
      element.setAttribute("name", nameInput.value.trim());
    }
    previewXml.value = serializeElement(element);
  });
  const nameLabel = document.createElement("label");
  nameLabel.textContent = "name attr";
  nameRow.appendChild(nameLabel);
  nameRow.appendChild(nameInput);
  configForm.appendChild(nameRow);

  const attrs = Array.from(element.attributes)
    .filter((attr) => attr.name !== "name")
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const attr of attrs) {
    const row = document.createElement("div");
    row.className = "config__row";
    const label = document.createElement("label");
    label.textContent = attr.name;
    const input = document.createElement("input");
    input.value = attr.value;
    input.addEventListener("input", () => {
      element.setAttribute(attr.name, input.value);
      previewXml.value = serializeElement(element);
    });
    row.appendChild(label);
    row.appendChild(input);
    configForm.appendChild(row);
  }

  const childElements = Array.from(element.children).filter((child) => child.children.length === 0);
  for (const child of childElements) {
    const row = document.createElement("div");
    row.className = "config__row";
    const label = document.createElement("label");
    label.textContent = child.tagName;
    const input = document.createElement("input");
    input.value = child.textContent || "";
    input.addEventListener("input", () => {
      child.textContent = input.value;
      previewXml.value = serializeElement(element);
    });
    row.appendChild(label);
    row.appendChild(input);
    configForm.appendChild(row);
  }
}

async function fetchCatalog() {
  const response = await fetch("/api/xml-builder/catalog");
  if (!response.ok) {
    throw new Error(`Catalog request failed: ${response.status}`);
  }
  return response.json();
}

async function fetchExampleXml(path) {
  const params = new URLSearchParams({ path });
  const response = await fetch(`/api/xml-builder/example?${params.toString()}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Example request failed: ${response.status}`);
  }
  return response.text();
}

async function fetchEntityOptions(entity) {
  if (!entity) {
    return [];
  }
  if (entityOptionCache.has(entity)) {
    return entityOptionCache.get(entity);
  }
  const params = new URLSearchParams({ entity });
  const response = await fetch(`/api/xml-builder/entity-options?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Entity options request failed: ${response.status}`);
  }
  const payload = await response.json();
  const options = payload.options || [];
  entityOptionCache.set(entity, options);
  return options;
}

function populateBaseSelect(examples) {
  baseSelect.innerHTML = "";
  const emptyOption = document.createElement("option");
  emptyOption.value = "";
  emptyOption.textContent = "Empty Simulation skeleton";
  baseSelect.appendChild(emptyOption);
  for (const example of examples) {
    const option = document.createElement("option");
    option.value = example.path;
    option.textContent = example.name;
    baseSelect.appendChild(option);
  }
}

async function loadBase() {
  const path = baseSelect.value;
  if (!path) {
    outputXml.value = skeletonXml();
    scheduleValidation();
    return;
  }
  const xmlText = await fetchExampleXml(path);
  outputXml.value = formatXml(xmlText);
  scheduleValidation();
}

function downloadXml() {
  const blob = new Blob([outputXml.value], { type: "application/xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "prlo_input.xml";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function copyXml() {
  await navigator.clipboard.writeText(outputXml.value);
  copyBtn.textContent = "Copied";
  setTimeout(() => {
    copyBtn.textContent = "Copy XML";
  }, 900);
}

function inferCurrentSection(xmlText, cursorIndex) {
  const sectionNames = new Set(ravenEntities);
  const prefix = xmlText.slice(0, cursorIndex);
  const tagPattern = /<\s*(\/?)([A-Za-z0-9_:-]+)([^>]*?)(\/?)\s*>/g;
  const stack = [];
  let match;
  while ((match = tagPattern.exec(prefix)) !== null) {
    const isClosing = match[1] === "/";
    const tagName = match[2];
    const isSelfClosing = match[4] === "/" || match[0].endsWith("/>");
    if (!sectionNames.has(tagName)) {
      continue;
    }
    if (isClosing) {
      if (stack.length > 0 && stack[stack.length - 1] === tagName) {
        stack.pop();
      } else {
        const idx = stack.lastIndexOf(tagName);
        if (idx >= 0) {
          stack.splice(idx, 1);
        }
      }
      continue;
    }
    if (!isSelfClosing) {
      stack.push(tagName);
    }
  }
  return stack.length > 0 ? stack[stack.length - 1] : null;
}

function currentTagPrefixInfo(xmlText, cursorIndex) {
  const before = xmlText.slice(0, cursorIndex);
  const lastLt = before.lastIndexOf("<");
  if (lastLt < 0) {
    return null;
  }
  const afterLt = before.slice(lastLt + 1);
  if (afterLt.includes(">")) {
    return null;
  }
  // Ignore closing tags or processing instructions
  if (afterLt.startsWith("/") || afterLt.startsWith("?") || afterLt.startsWith("!")) {
    return null;
  }
  // token ends at first whitespace or '>'
  const tokenMatch = afterLt.match(/^([A-Za-z0-9_:-]*)$/);
  if (!tokenMatch) {
    return null;
  }
  return {
    start: lastLt,
    end: cursorIndex,
    prefix: tokenMatch[1] || "",
  };
}

function openAutocomplete(defaultSection, tagPrefixInfo) {
  paletteMode = "entity";
  paletteContextEntity = defaultSection;
  paletteReplaceRange = tagPrefixInfo;

  const hint = defaultSection ? `Context: ${defaultSection}` : "Context: <Simulation>";
  paletteHint.textContent = hint;
  paletteSearch.value = tagPrefixInfo ? tagPrefixInfo.prefix : "";
  renderAutocompleteList();
  paletteDialog.showModal();
  paletteSearch.focus();
}

function closePalette() {
  paletteDialog.close();
}

function buildTopLevelEntityOptions() {
  return ravenEntities.map((entity) => ({
    tag: entity,
    description: "Top-level RAVEN input block.",
    template: `<${entity}>\n</${entity}>`,
  }));
}

function filterOptions(options, query) {
  const normalizedQuery = normalizeWhitespace(query).toLowerCase();
  if (!normalizedQuery) {
    return options;
  }
  return options.filter((option) => {
    const haystack = [option.tag, option.description]
      .map((item) => normalizeWhitespace(item).toLowerCase())
      .join(" ");
    return haystack.includes(normalizedQuery);
  });
}

function renderAutocompleteList() {
  paletteList.innerHTML = "";
  const query = paletteSearch.value;

  const section = paletteContextEntity;
  const promise = section ? fetchEntityOptions(section) : Promise.resolve(buildTopLevelEntityOptions());
  promise
    .then((options) => {
      paletteActiveOptions = filterOptions(options, query);
      const onActivate = (option, shouldInsert) => {
        if (!shouldInsert) {
          return;
        }
        const info = paletteReplaceRange;
        const baseIndent = currentLineIndent(outputXml.value, info ? info.start : outputXml.selectionStart);
        const lines = option.template.split("\n");
        const lastIdx = lines.length - 1;
        const indented = lines
          .map((line, idx) => {
            if (idx === 0) {
              return baseIndent + line;
            }
            if (idx === lastIdx) {
              return baseIndent + line;
            }
            return baseIndent + "  " + line;
          })
          .join("\n");
        insertText(outputXml, indented, info);
        scheduleValidation();
        logMessage("insert", `Inserted <${option.tag}>`);
        closePalette();
      };

      for (const option of paletteActiveOptions.slice(0, 400)) {
        paletteList.appendChild(createOptionItem(option, onActivate));
      }
      if (paletteActiveOptions.length === 0) {
        const empty = document.createElement("div");
        empty.className = "hint";
        empty.textContent = "No matching options.";
        paletteList.appendChild(empty);
      }
    })
    .catch((error) => {
      logMessage("error", error.message);
      const row = document.createElement("div");
      row.className = "hint";
      row.textContent = error.message;
      paletteList.appendChild(row);
    });
}

function renderPaletteList(defaultSection, query) {
  paletteList.innerHTML = "";
  const normalizedQuery = normalizeWhitespace(query).toLowerCase();
  let candidates = snippets;
  if (defaultSection) {
    const inSection = snippets.filter((snippet) => snippet.section === defaultSection);
    const outSection = snippets.filter((snippet) => snippet.section !== defaultSection);
    candidates = [...inSection, ...outSection];
  }
  const filtered = candidates.filter((snippet) => {
    if (!normalizedQuery) {
      return true;
    }
    const haystack = [
      snippet.section,
      snippet.tag,
      snippet.label,
      snippet.name,
      snippet.source,
    ]
      .map((item) => normalizeWhitespace(item).toLowerCase())
      .join(" ");
    return haystack.includes(normalizedQuery);
  });
  for (const snippet of filtered.slice(0, 250)) {
    paletteList.appendChild(
      createSnippetItem(snippet, (activatedSnippet, shouldInsert) => {
        selectSnippet(activatedSnippet.id);
        if (shouldInsert) {
          try {
            insertSnippetIntoOutput(activatedSnippet);
            logMessage("insert", `Inserted ${buildSnippetTitle(activatedSnippet)}`);
            closePalette();
          } catch (error) {
            logMessage("error", error.message);
            alert(error.message);
          }
        }
      })
    );
  }
  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "hint";
    empty.textContent = "No matching blocks.";
    paletteList.appendChild(empty);
  }
}

function validateOutput() {
  const text = outputXml.value;
  if (!text.trim()) {
    logMessage("warn", "Output XML is empty.");
    return false;
  }
  try {
    const doc = parseXml(text);
    if (doc.documentElement.tagName !== "Simulation") {
      logMessage("error", `Root tag must be <Simulation>, got <${doc.documentElement.tagName}>`);
      return false;
    }
    const sections = Array.from(doc.documentElement.children).map((node) => node.tagName);
    logMessage("ok", `Valid XML. Sections: ${sections.join(", ") || "(none)"}`);
    return true;
  } catch (error) {
    logMessage("error", `XML parse error: ${error.message}`);
    return false;
  }
}

function scheduleValidation() {
  if (plotTimer) {
    clearTimeout(plotTimer);
  }
  plotTimer = setTimeout(() => {
    syncWorkflowFromOutput();
    validateOutput();
  }, 400);
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (inQuotes && text[i + 1] === '"') {
        current += '"';
        i += 1;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }
    if (!inQuotes && (char === "," || char === "\n" || char === "\r")) {
      if (char === "\r" && text[i + 1] === "\n") {
        i += 1;
      }
      row.push(current);
      current = "";
      if (char !== ",") {
        if (row.length > 1 || (row.length === 1 && row[0] !== "")) {
          rows.push(row);
        }
        row = [];
      }
      continue;
    }
    current += char;
  }
  row.push(current);
  if (row.length > 1 || (row.length === 1 && row[0] !== "")) {
    rows.push(row);
  }
  return rows;
}

function populatePlotSelectors(columns) {
  plotX.innerHTML = "";
  plotY.innerHTML = "";
  for (const name of columns) {
    const optionX = document.createElement("option");
    optionX.value = name;
    optionX.textContent = name;
    plotX.appendChild(optionX);
    const optionY = document.createElement("option");
    optionY.value = name;
    optionY.textContent = name;
    plotY.appendChild(optionY);
  }
  if (columns.length > 1) {
    plotX.value = columns[0];
    plotY.value = columns[1];
  }
}

function getNumericColumn(rows, columns, name) {
  const idx = columns.indexOf(name);
  if (idx < 0) {
    return [];
  }
  const values = [];
  for (const row of rows) {
    const raw = row[idx];
    const number = Number(raw);
    if (!Number.isFinite(number)) {
      continue;
    }
    values.push(number);
  }
  return values;
}

function drawPlot(xValues, yValues) {
  const ctx = plotCanvas.getContext("2d");
  ctx.clearRect(0, 0, plotCanvas.width, plotCanvas.height);
  if (xValues.length === 0 || yValues.length === 0) {
    ctx.fillStyle = "rgba(166, 178, 199, 0.8)";
    ctx.fillText("Load a CSV and pick numeric columns.", 12, 20);
    return;
  }
  const count = Math.min(xValues.length, yValues.length, 5000);
  const xs = xValues.slice(0, count);
  const ys = yValues.slice(0, count);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 32;
  const w = plotCanvas.width - pad * 2;
  const h = plotCanvas.height - pad * 2;

  function scale(value, min, max, span) {
    if (max === min) {
      return 0.5 * span;
    }
    return ((value - min) / (max - min)) * span;
  }

  ctx.strokeStyle = "rgba(231, 237, 247, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + h);
  ctx.lineTo(pad + w, pad + h);
  ctx.stroke();

  ctx.fillStyle = "rgba(125, 211, 252, 0.75)";
  for (let i = 0; i < count; i += 1) {
    const x = pad + scale(xs[i], minX, maxX, w);
    const y = pad + h - scale(ys[i], minY, maxY, h);
    ctx.fillRect(x, y, 2, 2);
  }
}

function renderPlot() {
  if (plotRows.length === 0 || plotColumns.length === 0) {
    drawPlot([], []);
    plotMeta.textContent = "";
    return;
  }
  const xName = plotX.value;
  const yName = plotY.value;
  const xs = getNumericColumn(plotRows, plotColumns, xName);
  const ys = getNumericColumn(plotRows, plotColumns, yName);
  drawPlot(xs, ys);
  plotMeta.textContent = `Points: ${Math.min(xs.length, ys.length)} (showing up to 5000)`;
}

async function bootstrap() {
  outputXml.value = skeletonXml();
  catalog = await fetchCatalog();
  snippets = catalog.snippets || [];
  populateBaseSelect(catalog.examples || []);
  populateEntitySelect();
  render();
  renderMessages();
  scheduleValidation();
  applyDensity(loadDensity());
  workflow = loadWorkflow();
  renderWorkflow();
  renderToolbox();
  closeToolboxOverlay();
  setDashboardLink("");
  await loadRuns();
  if (refreshRunsBtn) {
    refreshRunsBtn.addEventListener("click", () => {
      loadRuns();
    });
  }
  if (runsFilter) {
    runsFilter.addEventListener("input", () => {
      applyRunsFilter();
    });
  }
  if (openRunBtn && runNameInput) {
    openRunBtn.addEventListener("click", () => {
      const value = runNameInput.value.trim();
      if (!value) {
        logMessage("warn", "Enter a run folder name.");
        return;
      }
      window.open(`/api/xml-builder/run-folder/${encodeURIComponent(value)}/dashboard`, "_blank", "noreferrer");
    });
  }
  if (openRunPathBtn && runPathInput) {
    openRunPathBtn.addEventListener("click", () => {
      openRunPathDashboard(runPathInput.value);
    });
  }
  if (runPathInput) {
    runPathInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        openRunPathDashboard(runPathInput.value);
      }
    });
  }
  if (browseRunPathBtn) {
    browseRunPathBtn.addEventListener("click", () => {
      openBrowseDialog();
    });
  }
  if (closeRunPathDialog) {
    closeRunPathDialog.addEventListener("click", () => {
      closeBrowseDialog();
    });
  }
  if (browseUpBtn) {
    browseUpBtn.addEventListener("click", () => {
      const target = browseUpBtn.dataset.target || "";
      if (!target) {
        return;
      }
      loadBrowseDirs(target);
    });
  }
  if (browseGoBtn && browsePathInput) {
    browseGoBtn.addEventListener("click", () => {
      const nextPath = browsePathInput.value.trim();
      loadBrowseDirs(nextPath);
    });
  }
  if (browseUseBtn) {
    browseUseBtn.addEventListener("click", () => {
      if (runPathInput) {
        runPathInput.value = browseCurrentPath;
        setRunPathHint(`Selected: ${browseCurrentPath}`);
      }
      closeBrowseDialog();
    });
  }
  if (browseOpenBtn) {
    browseOpenBtn.addEventListener("click", () => {
      if (runPathInput) {
        runPathInput.value = browseCurrentPath;
      }
      closeBrowseDialog();
      openRunPathDashboard(browseCurrentPath);
    });
  }
}

function renderToolbox() {
  if (!toolbox) {
    return;
  }
  toolbox.innerHTML = "";
  for (const item of toolboxItems) {
    const groupSectionChild =
      item.kind === "group"
        ? (item.children || []).find((child) => child && child.kind === "section" && child.section)
        : null;
    const el = document.createElement("div");
    el.className = "toolbox__item";
    el.draggable = item.kind !== "group" || Boolean(groupSectionChild);
    if (item.kind === "group") {
      el.title = groupSectionChild
        ? `Click to open ${item.label} tools · Drag to add <${groupSectionChild.section}>`
        : `Open ${item.label} tools`;
    } else {
      el.title = `Drag to insert ${item.label}`;
    }

    const icon = document.createElement("div");
    icon.className = "toolbox__icon";
    const img = document.createElement("img");
    img.alt = item.label;
    img.src = item.icon;
    icon.appendChild(img);

    const label = document.createElement("div");
    label.className = "toolbox__label";
    label.textContent = item.label;

    el.appendChild(icon);
    el.appendChild(label);

    if (item.kind === "group") {
      el.addEventListener("click", () => openToolboxGroupOverlay(item));
      if (groupSectionChild) {
        el.addEventListener("dblclick", () => {
          try {
            insertToolboxItemAndTrack(groupSectionChild);
            logMessage("insert", `Inserted <${groupSectionChild.section}>`);
          } catch (error) {
            logMessage("error", error.message);
            alert(error.message);
          }
        });
      }
      if (groupSectionChild) {
        el.addEventListener("dragstart", (event) => {
          event.dataTransfer.setData("application/prlo-toolbox", groupSectionChild.id);
          event.dataTransfer.effectAllowed = "copy";
          try {
            event.dataTransfer.setDragImage(img, 24, 24);
          } catch (_err) {
            // ignore if unsupported
          }
        });
      }
      toolbox.appendChild(el);
      continue;
    }

    el.addEventListener("click", () => previewToolboxItem(item));
    el.addEventListener("dblclick", () => {
      try {
        insertToolboxItemAndTrack(item);
        logMessage("insert", `Inserted ${item.label}`);
      } catch (error) {
        logMessage("error", error.message);
        alert(error.message);
      }
    });

    el.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("application/prlo-toolbox", item.id);
      event.dataTransfer.effectAllowed = "copy";
      try {
        event.dataTransfer.setDragImage(img, 24, 24);
      } catch (_err) {
        // ignore if unsupported
      }
    });

    toolbox.appendChild(el);
  }
}

function openToolboxGroupOverlay(group) {
  if (!toolboxOverlay || !toolboxOverlayTitle || !toolboxOverlayList) {
    return;
  }
  activeToolboxGroupId = group.id;
  toolboxOverlay.hidden = false;
  toolboxOverlayTitle.textContent = group.label;
  toolboxOverlayList.innerHTML = "";

  const items = [];
  for (const child of group.children || []) {
    items.push(child);
  }

  function itemTagName(item) {
    if (!item) {
      return null;
    }
    if (item.kind === "section") {
      return item.section || null;
    }
    try {
      return parseSnippetElement(stripXmlDeclaration(item.xml || "")).tagName;
    } catch (_err) {
      return null;
    }
  }

  const existingTags = new Set(items.map((item) => itemTagName(item)).filter(Boolean));

  if (group.dynamic && group.entity) {
    const loading = document.createElement("div");
    loading.className = "hint";
    loading.textContent = "Loading…";
    toolboxOverlayList.appendChild(loading);
    fetchEntityOptions(group.entity)
      .then((options) => {
        if (!toolboxOverlayList) {
          return;
        }
        if (activeToolboxGroupId !== group.id) {
          return;
        }
        toolboxOverlayList.innerHTML = "";
        const dynamicItems = [];
        for (const option of options || []) {
          const tag = option.tag;
          if (!tag || existingTags.has(tag)) {
            continue;
          }
          const id = dynamicToolId(group.entity, tag);
          const icon =
            (subtypeIconOverrides[group.entity] && subtypeIconOverrides[group.entity][tag]) ||
            group.icon;
          const tool = {
            id,
            label: tag,
            icon,
            kind: "child",
            section: group.entity,
            xml: option.template,
          };
          registerDynamicToolItem(tool);
          dynamicItems.push(tool);
        }
        const allItems = [...items, ...dynamicItems];
        for (const child of allItems) {
          toolboxOverlayList.appendChild(createToolboxOverlayRow(child));
        }
      })
      .catch((error) => {
        logMessage("warn", error.message);
        if (activeToolboxGroupId !== group.id) {
          return;
        }
        toolboxOverlayList.innerHTML = "";
        for (const child of items) {
          toolboxOverlayList.appendChild(createToolboxOverlayRow(child));
        }
        const row = document.createElement("div");
        row.className = "hint";
        row.textContent = `Could not load ${group.entity} subtypes.`;
        toolboxOverlayList.appendChild(row);
      });
    return;
  }

  for (const child of items) {
    toolboxOverlayList.appendChild(createToolboxOverlayRow(child));
  }
}

function createToolboxOverlayRow(child) {
  const row = document.createElement("div");
  row.className = "toolbox-overlay__item";
  row.draggable = true;
  row.title = child.kind === "section" ? `Drag to insert <${child.section}>` : `Drag to insert ${child.label}`;

    const img = document.createElement("img");
    img.alt = child.label;
    img.src = child.icon;

    const label = document.createElement("div");
    label.className = "toolbox-overlay__label";
    label.textContent = child.label;

    row.appendChild(img);
    row.appendChild(label);

    row.addEventListener("click", () => previewToolboxItem(child));
    row.addEventListener("dblclick", () => {
      try {
        insertToolboxItemAndTrack(child);
        logMessage("insert", `Inserted ${child.label}`);
      } catch (error) {
        logMessage("error", error.message);
        alert(error.message);
      }
    });

  row.addEventListener("dragstart", (event) => {
    event.dataTransfer.setData("application/prlo-toolbox", child.id);
    event.dataTransfer.effectAllowed = "copy";
    try {
      event.dataTransfer.setDragImage(img, 16, 16);
    } catch (_err) {
      // ignore
    }
  });
  return row;
}

function closeToolboxOverlay() {
  if (!toolboxOverlay) {
    return;
  }
  activeToolboxGroupId = null;
  toolboxOverlay.hidden = true;
}

function applyDensity(index) {
  const clamped = Math.max(0, Math.min(densityPresets.length - 1, Number(index)));
  const preset = densityPresets[clamped];
  document.documentElement.style.setProperty("--palette-list-gap", preset.gap);
  document.documentElement.style.setProperty("--palette-item-padding", preset.padding);
  document.documentElement.style.setProperty("--palette-title-font", preset.title);
  document.documentElement.style.setProperty("--palette-meta-font", preset.meta);
  if (densitySlider) {
    densitySlider.value = String(clamped);
  }
}

function loadDensity() {
  const stored = window.localStorage.getItem("prlo.paletteDensity");
  if (stored == null) {
    return 2;
  }
  const parsed = Number(stored);
  if (!Number.isFinite(parsed)) {
    return 2;
  }
  return parsed;
}

if (entitySelect) {
  entitySelect.addEventListener("change", () => {
    window.localStorage.setItem("prlo.entityFilter", entitySelect.value);
    render();
  });
}
if (densitySlider) {
  densitySlider.addEventListener("input", () => {
    const value = Number(densitySlider.value);
    applyDensity(value);
    window.localStorage.setItem("prlo.paletteDensity", String(value));
  });
}
if (toolboxOverlayBack) {
  toolboxOverlayBack.addEventListener("click", () => closeToolboxOverlay());
}
baseSelect.addEventListener("change", async () => {
  try {
    await loadBase();
  } catch (error) {
    logMessage("error", error.message);
    alert(error.message);
  }
});
insertBtn.addEventListener("click", () => {
  const snippet = currentSnippet();
  if (!snippet) {
    return;
  }
  try {
    insertSnippetIntoOutput(snippet);
    logMessage("insert", `Inserted ${buildSnippetTitle(snippet)}`);
  } catch (error) {
    logMessage("error", error.message);
    alert(error.message);
  }
});
copyBtn.addEventListener("click", async () => {
  try {
    await copyXml();
    logMessage("ok", "Copied output XML to clipboard.");
  } catch (error) {
    logMessage("error", error.message);
    alert(error.message);
  }
});
downloadBtn.addEventListener("click", () => downloadXml());
validateBtn.addEventListener("click", () => validateOutput());
if (runBtn) {
  runBtn.addEventListener("click", async () => {
    try {
      await startRun();
    } catch (error) {
      logMessage("error", error.message);
      setRunStatus(`Run error: ${error.message}`);
      alert(error.message);
    }
  });
}
clearMessagesBtn.addEventListener("click", () => {
  messages = [];
  renderMessages();
});
outputXml.addEventListener("input", () => scheduleValidation());
outputXml.addEventListener("dragover", (event) => {
  if (event.dataTransfer.types.includes("application/prlo-toolbox")) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }
});
outputXml.addEventListener("drop", (event) => {
  const id = event.dataTransfer.getData("application/prlo-toolbox");
  if (!id) {
    return;
  }
  const item = toolboxItemById(id);
  if (!item) {
    return;
  }
  event.preventDefault();
  try {
    insertToolboxItemAndTrack(item);
    logMessage("insert", `Inserted ${item.label}`);
  } catch (error) {
    logMessage("error", error.message);
    alert(error.message);
  }
});

if (workflowStrip) {
  workflowStrip.addEventListener("dragover", (event) => {
    if (event.dataTransfer.types.includes("application/prlo-toolbox")) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    }
  });
  workflowStrip.addEventListener("drop", (event) => {
    const id = event.dataTransfer.getData("application/prlo-toolbox");
    if (!id) {
      return;
    }
    const item = toolboxItemById(id);
    if (!item) {
      return;
    }
    event.preventDefault();
    try {
      insertToolboxItemAndTrack(item);
      logMessage("insert", `Inserted ${item.label}`);
    } catch (error) {
      logMessage("error", error.message);
      alert(error.message);
    }
  });
}
outputXml.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.code === "Space") {
    event.preventDefault();
    const section = inferCurrentSection(outputXml.value, outputXml.selectionStart);
    const info = currentTagPrefixInfo(outputXml.value, outputXml.selectionStart);
    openAutocomplete(section, info);
  }
  if (!event.ctrlKey && !event.metaKey && event.key === "<") {
    // Let the '<' be inserted first, then open autocomplete.
    setTimeout(() => {
      const cursor = outputXml.selectionStart;
      const section = inferCurrentSection(outputXml.value, cursor);
      const info = currentTagPrefixInfo(outputXml.value, cursor);
      if (info) {
        openAutocomplete(section, info);
      }
    }, 0);
  }
});

paletteSearch.addEventListener("input", () => {
  if (paletteMode === "entity") {
    renderAutocompleteList();
    return;
  }
  const section = inferCurrentSection(outputXml.value, outputXml.selectionStart);
  renderPaletteList(section, paletteSearch.value);
});
paletteDialog.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closePalette();
  }
  if (event.key === "Enter") {
    event.preventDefault();
    const first = paletteList.querySelector(".list__item");
    if (first) {
      first.dispatchEvent(new MouseEvent("dblclick"));
    }
  }
});

plotFile.addEventListener("change", async () => {
  const file = plotFile.files && plotFile.files[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  const rows = parseCsv(text);
  if (rows.length < 2) {
    logMessage("error", "CSV did not contain enough rows.");
    plotRows = [];
    plotColumns = [];
    renderPlot();
    return;
  }
  plotColumns = rows[0].map((name, idx) => normalizeWhitespace(name) || `col${idx}`);
  plotRows = rows.slice(1);
  populatePlotSelectors(plotColumns);
  renderPlot();
  logMessage("ok", `Loaded CSV with ${plotRows.length} rows.`);
});
plotX.addEventListener("change", () => renderPlot());
plotY.addEventListener("change", () => renderPlot());

bootstrap().catch((error) => {
  console.error(error);
  logMessage("error", error.message);
  alert(error.message);
});
