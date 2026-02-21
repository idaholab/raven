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
const themeToggle = document.getElementById("themeToggle");
const validateBtn = document.getElementById("validateBtn");
const runBtn = document.getElementById("runBtn");
const nextTodoBtn = document.getElementById("nextTodoBtn");
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
const workflowHint = document.querySelector(".workflow-strip__hint");
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

const themeKey = "raven-xml-builder-theme";

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

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(themeKey, theme);
  } catch (_err) {
    // ignore storage issues
  }
  if (themeToggle) {
    const label = theme === "light" ? "Dark mode" : "Light mode";
    themeToggle.innerHTML = `${themeIconSvg(theme)}${label}`;
  }
}

function themeIconSvg(theme) {
  if (theme === "dark") {
    return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="5" stroke="currentColor" stroke-width="2"/><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
  }
  return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><path d="M21 14.5A8.5 8.5 0 1 1 9.5 3a7 7 0 0 0 11.5 11.5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';
}

function initTheme() {
  let theme = "dark";
  try {
    const saved = localStorage.getItem(themeKey);
    if (saved) {
      theme = saved;
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
      theme = "light";
    }
  } catch (_err) {
    // ignore storage issues
  }
  setTheme(theme);
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      setTheme(next);
    });
  }
}

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
    xml: `<name>TODO_TEST_NAME</name>
<author>TODO_AUTHOR</author>
<created>TODO_DATE</created>
<description>TODO_DESCRIPTION</description>`,
  },
  {
    id: "runinfo",
    label: "RunInfo",
    icon: "/static/xml_builder/icons/runinfo.svg",
    kind: "section",
    section: "RunInfo",
    xml: `<!-- WorkingDir: REQUIRED - Directory where RAVEN executes -->
<WorkingDir>TODO_WORKING_DIR</WorkingDir>

<!-- Sequence: REQUIRED - Ordered list of step names to execute -->
<Sequence>TODO_STEP1, TODO_STEP2</Sequence>

<!-- Optional: Uncomment and fill as needed -->
<!-- <batchSize>1</batchSize> -->
<!-- <JobName>TODO_JOB_NAME</JobName> -->`,
  },
  {
    id: "variablegroups",
    label: "VariableGroups",
    icon: "/static/xml_builder/icons/variablegroups.svg",
    kind: "section",
    section: "VariableGroups",
    xml: `<!-- Define variable groups for reuse -->
<Group name="TODO_GROUP_NAME">TODO_VAR1, TODO_VAR2</Group>`,
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

function toolboxItemForSection(section) {
  if (!section) {
    return null;
  }
  const items = flattenToolboxItems(toolboxItems);
  const sectionItem = items.find((item) => item && item.kind === "section" && item.section === section);
  if (sectionItem) {
    return sectionItem;
  }
  return toolboxItems.find((item) => item && item.kind === "group" && item.entity === section) || null;
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
  if (workflowHint) {
    workflowHint.style.display = workflow.length ? "none" : "";
  }
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
  return `<?xml version="1.0" ?>
<Simulation>
  <!-- ============================================ -->
  <!-- REQUIRED: Run Configuration -->
  <!-- ============================================ -->
  <RunInfo>
    <!-- WorkingDir: REQUIRED - Directory where RAVEN executes -->
    <WorkingDir>TODO_WORKING_DIR</WorkingDir>

    <!-- Sequence: REQUIRED - Ordered list of step names to execute -->
    <Sequence>TODO_STEP1</Sequence>

    <!-- Optional: Uncomment and fill as needed -->
    <!-- <batchSize>1</batchSize> -->
  </RunInfo>

  <!-- ============================================ -->
  <!-- Files: Input/output files for the simulation -->
  <!-- ============================================ -->
  <Files>
    <!-- Example: <Input name="TODO_INPUT_NAME">TODO_FILE_PATH</Input> -->
  </Files>

  <!-- ============================================ -->
  <!-- Models: Code interfaces, ROMs, etc. -->
  <!-- ============================================ -->
  <Models>
    <!-- Example: <Code name="TODO_MODEL_NAME" subType="TODO_CODE_TYPE"></Code> -->
  </Models>

  <!-- ============================================ -->
  <!-- DataObjects: Where to store results -->
  <!-- ============================================ -->
  <DataObjects>
    <!-- Example: <PointSet name="TODO_DATASET_NAME"></PointSet> -->
  </DataObjects>

  <!-- ============================================ -->
  <!-- REQUIRED: Workflow Steps -->
  <!-- ============================================ -->
  <Steps>
    <!-- Example: <MultiRun name="TODO_STEP1"></MultiRun> -->
  </Steps>
</Simulation>
`;
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
  const sectionNode = ensureSection(outputDoc, item.section);

  if (item.kind === "section") {
    // Section insertion - no workflow marker needed

    // If the section has initial XML content, populate it
    if (item.xml) {
      const templateXml = stripXmlDeclaration(item.xml);
      // Parse template and extract children
      const tempDoc = parseXml(`<Temp>${templateXml}</Temp>`);
      const children = Array.from(tempDoc.documentElement.childNodes);
      for (const child of children) {
        const imported = outputDoc.importNode(child, true);
        sectionNode.appendChild(imported);
      }
    }

    outputXml.value = serializeXml(outputDoc);
    scheduleValidation();
    return;
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
  const imported = outputDoc.importNode(blockNode, true);
  sectionNode.appendChild(imported);
  outputXml.value = serializeXml(outputDoc);
  scheduleValidation();
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
      return;
    }
    if (workflow.length === 0) {
      const inferred = [];
      for (const child of Array.from(root.children)) {
        if (!child || child.nodeType !== 1) {
          continue;
        }
        const tool = toolboxItemForSection(child.tagName);
        if (tool) {
          inferred.push({ instanceId: null, toolId: tool.id });
        }
      }
      if (inferred.length > 0) {
        workflow = inferred;
        persistWorkflow();
        renderWorkflow();
      }
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
    outputXml.dispatchEvent(new Event('input', { bubbles: true }));
    scheduleValidation();
    return;
  }
  const xmlText = await fetchExampleXml(path);
  outputXml.value = formatXml(xmlText);
  outputXml.dispatchEvent(new Event('input', { bubbles: true }));
  scheduleValidation();
}

function jumpToNextTodo() {
  const text = outputXml.value;
  const currentPos = outputXml.selectionStart;

  // Search for TODO patterns: "TODO", "TODO_NAME", "TODO: Fill in", etc.
  const todoPattern = /TODO[_:]?[A-Z_]*/gi;

  // Find all TODO occurrences
  let match;
  const todos = [];
  while ((match = todoPattern.exec(text)) !== null) {
    todos.push({
      index: match.index,
      text: match[0],
      length: match[0].length
    });
  }

  if (todos.length === 0) {
    logMessage("info", "No TODO placeholders found in the XML.");
    return;
  }

  // Find the next TODO after current cursor position
  const nextTodo = todos.find(todo => todo.index > currentPos);

  if (nextTodo) {
    // Found a TODO after cursor - jump to it
    outputXml.focus();
    outputXml.setSelectionRange(nextTodo.index, nextTodo.index + nextTodo.length);
    outputXml.scrollTop = Math.max(0,
      (outputXml.scrollHeight / text.length) * nextTodo.index - outputXml.clientHeight / 2
    );
    logMessage("ok", `Jumped to: ${nextTodo.text}`);
  } else {
    // No TODO after cursor - wrap around to first one
    const firstTodo = todos[0];
    outputXml.focus();
    outputXml.setSelectionRange(firstTodo.index, firstTodo.index + firstTodo.length);
    outputXml.scrollTop = Math.max(0,
      (outputXml.scrollHeight / text.length) * firstTodo.index - outputXml.clientHeight / 2
    );
    logMessage("ok", `Wrapped to first TODO: ${firstTodo.text}`);
  }
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
  const prefix = xmlText.slice(0, cursorIndex);
  const tagPattern = /<\s*(\/?)([A-Za-z0-9_:-]+)([^>]*?)(\/?)\s*>/g;
  const stack = [];
  let match;
  while ((match = tagPattern.exec(prefix)) !== null) {
    const isClosing = match[1] === "/";
    const tagName = match[2];
    const isSelfClosing = match[4] === "/" || match[0].endsWith("/>");

    // Skip XML declarations and non-entity tags, but track all structural tags
    if (tagName === "Simulation" || tagName === "?xml") {
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
  // Return the most specific context (last item in stack)
  // This will be the innermost tag like "MultiRun" instead of just "Steps"
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

  // Add compact autocomplete styling
  paletteDialog.classList.add('dialog--autocomplete');

  // Position at cursor
  if (outputXml) {
    try {
      // Get cursor position in the textarea
      const cursorPos = outputXml.selectionStart;
      const text = outputXml.value;
      const textBeforeCursor = text.substring(0, cursorPos);
      const lines = textBeforeCursor.split('\n');
      const currentLine = lines.length;
      const currentCol = lines[lines.length - 1].length;

      // Calculate approximate pixel position
      // Note: This is an approximation based on font metrics
      const lineHeight = 16.8; // 12px font-size * 1.4 line-height
      const charWidth = 7.2; // Approximate monospace character width
      const padding = 12;

      const textareaRect = outputXml.getBoundingClientRect();
      const scrollTop = outputXml.scrollTop;
      const scrollLeft = outputXml.scrollLeft;

      const top = textareaRect.top + (currentLine - 1) * lineHeight + padding - scrollTop + lineHeight;
      const left = textareaRect.left + currentCol * charWidth + padding - scrollLeft;

      // Constrain to viewport
      const maxTop = window.innerHeight - 250; // Leave room for dialog
      const maxLeft = window.innerWidth - 220; // Leave room for dialog width

      const finalTop = Math.min(Math.max(top, 50), maxTop);
      const finalLeft = Math.min(Math.max(left, 20), maxLeft);

      paletteDialog.style.setProperty('--autocomplete-top', `${finalTop}px`);
      paletteDialog.style.setProperty('--autocomplete-left', `${finalLeft}px`);
      paletteDialog.style.transform = 'none'; // Don't center, use exact position
    } catch (err) {
      // Fallback to center if positioning fails
      paletteDialog.style.transform = '';
    }
  }

  renderAutocompleteList();
  paletteDialog.showModal();
  paletteSearch.focus();
}

function closePalette() {
  paletteDialog.classList.remove('dialog--autocomplete');
  paletteDialog.style.transform = '';
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
        const insertStart = info ? info.start : outputXml.selectionStart;
        const baseIndent = currentLineIndent(outputXml.value, insertStart);
        const lines = option.template.split("\n");
        const indented = lines
          .map((line, idx) => {
            if (idx === 0) {
              // First line: don't add baseIndent (it's already in the text before the replacement)
              return line;
            }
            // All subsequent lines: add baseIndent to match the parent's indentation
            // (templates already have their own relative indentation built-in)
            return baseIndent + line;
          })
          .join("\n");
        insertText(outputXml, indented, info);

        // Jump to first TODO placeholder after insertion for easy editing
        const textAfter = outputXml.value.substring(insertStart);
        const todoMatch = textAfter.match(/TODO[_:]?[A-Z_]*/);
        if (todoMatch) {
          const todoStart = insertStart + todoMatch.index;
          const todoEnd = todoStart + todoMatch[0].length;
          outputXml.setSelectionRange(todoStart, todoEnd);
        }

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
    updateSyntaxHighlighting();
    autoCreateReferencedEntities();
    // Note: updateGutter() is called separately via input event listener
  }, 400);
}

function groupIdForSection(section) {
  if (!section) {
    return null;
  }
  const group = toolboxItems.find((item) => item && item.kind === "group" && item.entity === section);
  return group ? group.id : null;
}

function resolveSubtypeToolId(section, tag) {
  if (!section || !tag) {
    return null;
  }
  for (const item of flattenToolboxItems(toolboxItems)) {
    if (!item || item.kind !== "child" || item.section !== section || !item.xml) {
      continue;
    }
    try {
      const tagName = parseSnippetElement(stripXmlDeclaration(item.xml)).tagName;
      if (tagName === tag) {
        return item.id;
      }
    } catch (_err) {
      // ignore malformed snippet
    }
  }
  const dynId = dynamicToolId(section, tag);
  toolboxItemById(dynId);
  return dynId;
}

/**
 * Auto-create referenced entities when they're mentioned in Steps
 * For example, if a step references a Sampler, auto-create the Samplers section
 */
function autoCreateReferencedEntities() {
  try {
    const outputDoc = parseXml(outputXml.value);
    const root = outputDoc.documentElement;
    if (!root || root.tagName !== "Simulation") {
      return;
    }

    // Find the Steps section
    const stepsNode = Array.from(root.children).find(child => child.tagName === "Steps");
    if (!stepsNode) {
      return;
    }

    // Map of step child elements to their corresponding entity sections
    const stepToEntityMap = {
      'Sampler': 'Samplers',
      'Optimizer': 'Optimizers',
      'Model': 'Models',
      'DataObject': 'DataObjects',
      'Output': 'OutStreams'
    };

    // Find all step children that reference entities
    const referencedEntities = [];
    for (const step of stepsNode.children) {
      if (step.nodeType !== 1) continue; // Element nodes only

      for (const child of step.children) {
        if (child.nodeType !== 1) continue;

        const childTag = child.tagName;
        const classAttr = safeText(child.getAttribute("class"));
        const typeAttr = safeText(child.getAttribute("type"));
        const entitySection = classAttr || stepToEntityMap[childTag];
        const entityTag = typeAttr || childTag;
        if (!entitySection) {
          continue;
        }
        const refName = child.textContent.trim();
        if (refName && !refName.startsWith('TODO')) {
          referencedEntities.push({
            type: entityTag,
            name: refName,
            section: entitySection
          });
        }
      }
    }

    if (referencedEntities.length === 0) {
      return;
    }

    // Check which sections need to be created
    let needsUpdate = false;
    const entitiesToCreate = [];

    for (const ref of referencedEntities) {
      // Check if section exists
      let sectionNode = Array.from(root.children).find(child => child.tagName === ref.section);

      if (!sectionNode) {
        // Section doesn't exist, we need to create it
        entitiesToCreate.push({
          ...ref,
          createSection: true,
          createEntity: true
        });
        needsUpdate = true;
      } else {
        // Section exists, check if the entity with this name exists
        const existingEntity = Array.from(sectionNode.children).find(
          child => child.nodeType === 1 &&
                   child.tagName === ref.type &&
                   child.getAttribute('name') === ref.name
        );

        if (!existingEntity) {
          entitiesToCreate.push({
            ...ref,
            createSection: false,
            createEntity: true
          });
          needsUpdate = true;
        }
      }
    }

    if (!needsUpdate) {
      return;
    }

    // Collect sections that need animation and deduplicate
    const sectionsToAnimate = [];
    const createdSections = new Set();
    const sectionToToolId = {
      DataObjects: "dataobjects",
      Models: "models",
      Optimizers: "optimizers",
      OutStreams: "outstreams",
      Samplers: "samplers"
    };
    const animatedToolIds = new Set();

    // Create missing sections and entities
    for (const item of entitiesToCreate) {
      if (item.createSection && !createdSections.has(item.section)) {
        // Section hasn't been created yet in this run
        createdSections.add(item.section);

        // Find the right position to insert the section (before Steps)
        const stepsNode = Array.from(root.children).find(child => child.tagName === "Steps");
        const sectionNode = outputDoc.createElement(item.section);

        root.insertBefore(outputDoc.createComment(` ${item.section} `), stepsNode);
        root.insertBefore(sectionNode, stepsNode);

        // Add the entity to the new section
        const entityNode = outputDoc.createElement(item.type);
        entityNode.setAttribute('name', item.name);
        entityNode.appendChild(outputDoc.createComment(' TODO: Configure this '));
        sectionNode.appendChild(entityNode);

        // Mark for animation
        const groupId = sectionToToolId[item.section] || groupIdForSection(item.section);
        const subtypeToolId = resolveSubtypeToolId(item.section, item.type);
        const animateId = subtypeToolId || groupId;
        if (animateId && !animatedToolIds.has(animateId)) {
          animatedToolIds.add(animateId);
          sectionsToAnimate.push({
            toolId: animateId,
            groupId,
            subtypeTag: item.type,
            entitySection: item.section
          });
        }
      } else if (item.createSection && createdSections.has(item.section)) {
        // Section was already created in this run, just add the entity
        const sectionNode = Array.from(root.children).find(child => child.tagName === item.section);
        const entityNode = outputDoc.createElement(item.type);
        entityNode.setAttribute('name', item.name);
        entityNode.appendChild(outputDoc.createComment(' TODO: Configure this '));
        sectionNode.appendChild(entityNode);
      } else if (item.createEntity) {
        // Section exists, just add the entity
        const sectionNode = Array.from(root.children).find(child => child.tagName === item.section);
        const entityNode = outputDoc.createElement(item.type);
        entityNode.setAttribute('name', item.name);
        entityNode.appendChild(outputDoc.createComment(' TODO: Configure this '));
        sectionNode.appendChild(entityNode);
        const groupId = sectionToToolId[item.section] || groupIdForSection(item.section);
        const subtypeToolId = resolveSubtypeToolId(item.section, item.type);
        const animateId = subtypeToolId || groupId;
        if (animateId && !animatedToolIds.has(animateId)) {
          animatedToolIds.add(animateId);
          sectionsToAnimate.push({
            toolId: animateId,
            groupId,
            subtypeTag: item.type,
            entitySection: item.section
          });
        }
      }
    }

    // Update the output
    outputXml.value = serializeXml(outputDoc);
    outputXml.dispatchEvent(new Event('input', { bubbles: true }));

    if (entitiesToCreate.length > 0) {
      logMessage("ok", `Auto-created ${entitiesToCreate.length} referenced entit${entitiesToCreate.length === 1 ? 'y' : 'ies'}`);
    }

    // Trigger animations after a brief delay to ensure DOM is updated
    if (sectionsToAnimate.length > 0) {
      setTimeout(() => {
        sectionsToAnimate.forEach((item, index) => {
          // Stagger animations slightly for multiple items
          setTimeout(() => {
            animateToolboxToWorkflow(item);
          }, index * 800);
        });
      }, 200);
    }
  } catch (err) {
    // Ignore errors during auto-creation
  }
}

function prefersReducedMotion() {
  return Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function waitForOverlayItem(toolId, timeoutMs = 2500) {
  return new Promise((resolve) => {
    const start = Date.now();
    const check = () => {
      const el = document.querySelector(`.toolbox-overlay__item[data-id="${toolId}"]`);
      if (el) {
        resolve(el);
        return;
      }
      if (Date.now() - start >= timeoutMs) {
        resolve(null);
        return;
      }
      setTimeout(check, 60);
    };
    check();
  });
}

function isItemVisibleInToolbox(toolbox, item, vertical) {
  const container = toolbox.getBoundingClientRect();
  const rect = item.getBoundingClientRect();
  if (vertical) {
    return rect.top >= container.top && rect.bottom <= container.bottom;
  }
  return rect.left >= container.left && rect.right <= container.right;
}

function computeTargetScroll(toolbox, item, vertical) {
  const container = toolbox.getBoundingClientRect();
  const rect = item.getBoundingClientRect();
  if (vertical) {
    const target = toolbox.scrollTop + (rect.top - container.top) - (container.height - rect.height) / 2;
    const max = toolbox.scrollHeight - toolbox.clientHeight;
    return clamp(target, 0, Math.max(0, max));
  }
  const target = toolbox.scrollLeft + (rect.left - container.left) - (container.width - rect.width) / 2;
  const max = toolbox.scrollWidth - toolbox.clientWidth;
  return clamp(target, 0, Math.max(0, max));
}

function createRavenElement() {
  const raven = document.createElement("div");
  raven.className = "raven raven--fly";
  raven.setAttribute("aria-hidden", "true");
  raven.innerHTML = `
    <img
      class="raven__img"
      alt=""
      src="/static/xml_builder/icons/raven.png"
    />
  `;
  return raven;
}

async function animateToolboxToWorkflow(target) {
  try {
    const toolId = typeof target === "string" ? target : target && target.toolId;
    const groupId = typeof target === "string" ? null : target && target.groupId;
    const subtypeTag = typeof target === "string" ? null : target && target.subtypeTag;
    const entitySection = typeof target === "string" ? null : target && target.entitySection;

    const groupToolId = groupId || groupIdForSection(entitySection) || toolId;
    if (!groupToolId) {
      return;
    }

    const toolboxItems = document.querySelectorAll(".toolbox__item");
    let toolboxItem = null;
    for (const item of toolboxItems) {
      if (item.getAttribute("data-id") === groupToolId) {
        toolboxItem = item;
        break;
      }
    }

    if (!toolboxItem) {
      return;
    }

    const toolbox = document.getElementById("toolbox");
    const workflowStrip = document.getElementById("workflowStrip");
    if (!toolbox || !workflowStrip) {
      return;
    }

    if (prefersReducedMotion()) {
      toolboxItem.scrollIntoView({ block: "nearest", inline: "nearest" });
      toolboxItem.classList.add("toolbox__item--picked");
      setTimeout(() => toolboxItem.classList.remove("toolbox__item--picked"), 600);
      return;
    }

    const vertical = toolbox.classList.contains("toolbox--vertical");
    const raven = createRavenElement();
    document.body.appendChild(raven);

    const ravenRect = raven.getBoundingClientRect();
    const toolboxRect = toolbox.getBoundingClientRect();

    const placeRaven = (x, y, duration = 0) => {
      raven.style.transition = duration
        ? `transform ${duration}ms cubic-bezier(0.2, 0.8, 0.2, 1)`
        : "none";
      raven.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    };

    const centerRavenOn = (x, y) => {
      return {
        x: x - ravenRect.width / 2,
        y: y - ravenRect.height / 2
      };
    };

    const startPoint = centerRavenOn(toolboxRect.left + 20, toolboxRect.top + 40);
    placeRaven(startPoint.x, startPoint.y);
    await wait(50);

    const scrollbarPoint = vertical
      ? centerRavenOn(toolboxRect.right - 8, toolboxRect.top + toolboxRect.height * 0.4)
      : centerRavenOn(toolboxRect.left + toolboxRect.width * 0.6, toolboxRect.bottom - 8);

    placeRaven(scrollbarPoint.x, scrollbarPoint.y, 420);
    await wait(450);

    const targetScroll = computeTargetScroll(toolbox, toolboxItem, vertical);
    const scrollAxis = vertical ? "scrollTop" : "scrollLeft";
    const maxScroll = vertical
      ? toolbox.scrollHeight - toolbox.clientHeight
      : toolbox.scrollWidth - toolbox.clientWidth;

    const scrollDistance = Math.abs(targetScroll - toolbox[scrollAxis]);
    if (maxScroll > 4 && scrollDistance > 4) {
      const stepCount = clamp(Math.ceil(scrollDistance / 70), 3, 10);
      const stepDelta = (targetScroll - toolbox[scrollAxis]) / stepCount;
      for (let i = 0; i < stepCount; i += 1) {
        raven.classList.add("raven--peck");
        toolbox[scrollAxis] = clamp(toolbox[scrollAxis] + stepDelta, 0, maxScroll);
        await wait(180);
        raven.classList.remove("raven--peck");
        await wait(110);
      }
    }

    if (!isItemVisibleInToolbox(toolbox, toolboxItem, vertical)) {
      toolboxItem.scrollIntoView({ block: "nearest", inline: "nearest" });
      await wait(150);
    }

    let pickupElement = toolboxItem;
    if (subtypeTag || (toolId && toolId !== groupToolId)) {
      const group = toolboxItemById(groupToolId);
      if (group && group.kind === "group") {
        openToolboxGroupOverlay(group);
        const desiredToolId = toolId || resolveSubtypeToolId(entitySection || group.entity, subtypeTag);
        if (desiredToolId) {
          const overlayElement = await waitForOverlayItem(desiredToolId);
          if (overlayElement) {
            pickupElement = overlayElement;
          }
        }
      }
    }

    const itemRect = pickupElement.getBoundingClientRect();
    const itemPoint = centerRavenOn(itemRect.left + itemRect.width * 0.55, itemRect.top + itemRect.height * 0.25);
    placeRaven(itemPoint.x, itemPoint.y, 380);
    await wait(420);

    toolboxItem.classList.add("toolbox__item--picked");
    raven.classList.add("raven--snatch");
    await wait(240);
    raven.classList.remove("raven--snatch");

    const cargo = pickupElement.cloneNode(true);
    cargo.classList.add("raven__cargo");
    cargo.setAttribute("aria-hidden", "true");
    raven.appendChild(cargo);

    const workflowRect = workflowStrip.getBoundingClientRect();
    const workflowPoint = centerRavenOn(
      workflowRect.left + workflowRect.width * 0.6,
      workflowRect.top + workflowRect.height * 0.35
    );
    placeRaven(workflowPoint.x, workflowPoint.y, 620);
    await wait(640);

    cargo.classList.add("raven__cargo--drop");
    await wait(280);

    toolboxItem.classList.remove("toolbox__item--picked");
    raven.classList.add("raven--exit");
    placeRaven(workflowPoint.x + 140, workflowPoint.y - 120, 520);
    await wait(540);

    raven.remove();
  } catch (err) {
    console.error("Animation error:", err);
  }
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
  // Trigger input event to update gutter and highlighting
  outputXml.dispatchEvent(new Event('input', { bubbles: true }));
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
    el.setAttribute('data-id', item.id); // Add data-id for animation
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
  row.setAttribute("data-id", child.id);

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
if (nextTodoBtn) {
  nextTodoBtn.addEventListener("click", () => jumpToNextTodo());
}
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
    event.preventDefault();
    if (event.dataTransfer) {
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

// XML Syntax Highlighting
const outputXmlHighlight = document.getElementById("outputXmlHighlight");

function highlightXML(xmlText) {
  if (!xmlText) {
    return "";
  }

  // Escape HTML entities first
  const escapeHtml = (text) => {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  let highlighted = escapeHtml(xmlText);

  // Highlight TODO markers
  highlighted = highlighted.replace(/TODO[_A-Z0-9]*/g, (match) => {
    return `<span class="xml-todo">${match}</span>`;
  });

  // Hide fold marker comments (make them invisible)
  highlighted = highlighted.replace(/&lt;!--FOLD:[^>]+--&gt;/g, () => {
    return '<span style="display:none;"></span>';
  });

  // Highlight XML comments (but not fold markers which are already handled)
  highlighted = highlighted.replace(/&lt;!--(.*?)--&gt;/g, (_match, content) => {
    return `<span class="xml-comment">&lt;!--${content}--&gt;</span>`;
  });

  // Highlight XML tags with attributes
  highlighted = highlighted.replace(/(&lt;\/?)([\w:]+)(.*?)(\/?&gt;)/g, (match, openBracket, tagName, attrs, closeBracket) => {
    // Skip if this is part of a comment
    if (match.includes('xml-comment')) {
      return match;
    }

    let highlightedAttrs = attrs;

    // Highlight attributes: name="value"
    highlightedAttrs = highlightedAttrs.replace(/([\w:]+)(=)(&quot;)(.*?)(&quot;)/g, (_attrMatch, attrName, equals, openQuote, attrValue, closeQuote) => {
      // Check if attribute value contains TODO
      let highlightedValue = attrValue;
      if (attrValue.includes('TODO')) {
        highlightedValue = attrValue.replace(/TODO[_A-Z0-9]*/g, (todoMatch) => {
          return `<span class="xml-todo">${todoMatch}</span>`;
        });
      }
      return `<span class="xml-attr-name">${attrName}</span>${equals}${openQuote}<span class="xml-attr-value">${highlightedValue}</span>${closeQuote}`;
    });

    return `${openBracket}<span class="xml-tag">${tagName}</span>${highlightedAttrs}${closeBracket}`;
  });

  return highlighted;
}

function updateSyntaxHighlighting() {
  if (outputXmlHighlight && outputXml) {
    outputXmlHighlight.innerHTML = highlightXML(outputXml.value);
    // Set height to match the full scrollable content height
    outputXmlHighlight.style.height = outputXml.scrollHeight + 'px';
  }
}

function syncScroll() {
  if (outputXmlHighlight && outputXml) {
    const scrollTop = outputXml.scrollTop;
    const scrollLeft = outputXml.scrollLeft;
    outputXmlHighlight.style.transform = `translate(-${scrollLeft}px, -${scrollTop}px)`;
  }
}

if (outputXml && outputXmlHighlight) {
  // Update highlighting on input
  outputXml.addEventListener("input", updateSyntaxHighlighting);

  // Sync scrolling
  outputXml.addEventListener("scroll", syncScroll);

  // Tab key support for indentation
  outputXml.addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
      e.preventDefault();

      const start = outputXml.selectionStart;
      const end = outputXml.selectionEnd;
      const value = outputXml.value;

      if (e.shiftKey) {
        // Shift+Tab: Unindent
        const lineStart = value.lastIndexOf("\n", start - 1) + 1;
        const lineEnd = value.indexOf("\n", end);
        const endPos = lineEnd === -1 ? value.length : lineEnd;

        const selectedLines = value.substring(lineStart, endPos);
        const unindentedLines = selectedLines.split("\n").map(line => {
          // Remove up to 2 spaces from the start of each line
          if (line.startsWith("  ")) {
            return line.substring(2);
          } else if (line.startsWith(" ")) {
            return line.substring(1);
          }
          return line;
        }).join("\n");

        outputXml.value = value.substring(0, lineStart) + unindentedLines + value.substring(endPos);
        outputXml.selectionStart = start - (start > lineStart && value.charAt(lineStart) === " " ? (value.charAt(lineStart + 1) === " " ? 2 : 1) : 0);
        outputXml.selectionEnd = lineStart + unindentedLines.length;
      } else {
        // Tab: Indent
        if (start === end) {
          // No selection - insert 2 spaces at cursor
          outputXml.value = value.substring(0, start) + "  " + value.substring(end);
          outputXml.selectionStart = outputXml.selectionEnd = start + 2;
        } else {
          // Selection - indent all selected lines
          const lineStart = value.lastIndexOf("\n", start - 1) + 1;
          const lineEnd = value.indexOf("\n", end);
          const endPos = lineEnd === -1 ? value.length : lineEnd;

          const selectedLines = value.substring(lineStart, endPos);
          const indentedLines = selectedLines.split("\n").map(line => "  " + line).join("\n");

          outputXml.value = value.substring(0, lineStart) + indentedLines + value.substring(endPos);
          outputXml.selectionStart = start + 2;
          outputXml.selectionEnd = lineStart + indentedLines.length;
        }
      }

      updateSyntaxHighlighting();
      scheduleValidation();
    }
  });

  // Initial highlighting
  updateSyntaxHighlighting();
}

// ============================================================================
// Code Folding
// ============================================================================

const codeGutter = document.getElementById("codeGutter");

/**
 * Find foldable regions in the XML text
 * Returns array of {startLine, endLine, tagName, indent}
 */
function findFoldableRegions(text) {
  const lines = text.split("\n");
  const regions = [];
  const stack = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip comments and empty lines
    if (trimmed.startsWith("<!--") || trimmed === "") {
      continue;
    }

    // Match opening tag: <TagName ...> but not self-closing <... /> or closing </...>
    const openMatch = trimmed.match(/^<([a-zA-Z][\w:.-]*)[^>]*>$/);
    if (openMatch && !trimmed.endsWith("/>") && !trimmed.startsWith("</") && !trimmed.startsWith("<?")) {
      const tagName = openMatch[1];
      const indent = line.length - line.trimStart().length;
      stack.push({ tagName, startLine: i, indent });
      continue;
    }

    // Match closing tag: </TagName>
    const closeMatch = trimmed.match(/^<\/([a-zA-Z][\w:.-]*)>$/);
    if (closeMatch) {
      const tagName = closeMatch[1];
      // Find matching opening tag from stack
      for (let j = stack.length - 1; j >= 0; j--) {
        if (stack[j].tagName === tagName) {
          const opening = stack.splice(j, 1)[0];
          // Only create fold region if there's content between tags (more than 1 line)
          if (i - opening.startLine > 0) {
            regions.push({
              startLine: opening.startLine,
              endLine: i,
              tagName: tagName,
              indent: opening.indent
            });
          }
          break;
        }
      }
    }
  }

  return regions;
}

/**
 * Render the gutter with line numbers and fold indicators
 */
function updateGutter() {
  if (!codeGutter || !outputXml) return;

  const text = outputXml.value;
  const lines = text.split("\n");
  const regions = findFoldableRegions(text);

  // Create a map of startLine -> region for quick lookup
  const regionMap = new Map();
  regions.forEach(region => {
    regionMap.set(region.startLine, region);
  });

  // Build gutter HTML
  let gutterHTML = "";
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const region = regionMap.get(i);

    // Check if this line is folded (contains the fold marker)
    const isFolded = line.includes("⋯");

    gutterHTML += '<div class="gutter-line">';

    // Add fold indicator if this line starts a foldable region OR is a folded line
    if (region || isFolded) {
      const foldIcon = isFolded ? "▸" : "▾";
      gutterHTML += `<span class="gutter-line__fold" data-line="${i}" title="${isFolded ? 'Unfold' : 'Fold'}">${foldIcon}</span>`;
    }

    // Line number
    const lineNum = i + 1;
    gutterHTML += `<span class="gutter-line__number">${lineNum}</span>`;

    gutterHTML += '</div>';
  }

  codeGutter.innerHTML = gutterHTML;

  // Attach click handlers to fold indicators
  codeGutter.querySelectorAll(".gutter-line__fold").forEach(fold => {
    fold.addEventListener("click", handleFoldClick);
  });

  // Sync scroll
  syncGutterScroll();
}

/**
 * Handle click on fold indicator
 */
function handleFoldClick(e) {
  const lineNum = parseInt(e.target.getAttribute("data-line"), 10);

  if (!outputXml) return;

  const lines = outputXml.value.split("\n");
  const line = lines[lineNum];

  // Check if the line is currently folded (contains ⋯)
  if (line && line.includes("⋯")) {
    // Unfold
    unfoldRegion(lineNum);
  } else {
    // Fold
    foldRegion(lineNum);
  }
}

/**
 * Fold a region starting at the given line
 */
function foldRegion(startLine) {
  if (!outputXml) return;

  const text = outputXml.value;
  const regions = findFoldableRegions(text);
  const region = regions.find(r => r.startLine === startLine);

  if (!region) return;

  const lines = text.split("\n");
  const startLineText = lines[region.startLine];
  const endLineText = lines[region.endLine];

  // Store the original folded content using opening tag as key
  const foldedContent = lines.slice(region.startLine + 1, region.endLine).join("\n");
  if (!window.foldedContentMap) {
    window.foldedContentMap = new Map();
  }
  const openingTag = startLineText.trim();
  const storageKey = `${openingTag}_${region.startLine}`;
  window.foldedContentMap.set(storageKey, {
    content: foldedContent,
    openingTag: openingTag,
    closingTag: endLineText.trim()
  });

  // Create folded text: opening tag with "..." and closing tag on same line
  // Include storage key as a hidden marker
  const closingTag = endLineText.trim();
  const indentation = startLineText.substring(0, startLineText.length - openingTag.length);
  const foldedLine = `${indentation}${openingTag} ⋯ ${closingTag} <!--FOLD:${storageKey}-->`;

  // Reconstruct the text with the folded region
  const newLines = [
    ...lines.slice(0, region.startLine),
    foldedLine,
    ...lines.slice(region.endLine + 1)
  ];

  // Update textarea
  const cursorPos = outputXml.selectionStart;
  outputXml.value = newLines.join("\n");

  // Restore cursor position (approximately)
  outputXml.selectionStart = outputXml.selectionEnd = Math.min(cursorPos, outputXml.value.length);

  // Update UI
  updateSyntaxHighlighting();
  updateGutter();
  scheduleValidation();
}

/**
 * Unfold a region at the given line (which should be a folded line)
 */
function unfoldRegion(foldedLineIndex) {
  if (!outputXml || !window.foldedContentMap) return;

  const text = outputXml.value;
  const lines = text.split("\n");
  const foldedLine = lines[foldedLineIndex];

  if (!foldedLine || !foldedLine.includes("⋯")) {
    return; // Not a folded line
  }

  // Extract the storage key from the comment marker
  const keyMatch = foldedLine.match(/<!--FOLD:(.*?)-->/);
  let storageKey = null;
  let foldData = null;

  if (keyMatch) {
    storageKey = keyMatch[1];
    foldData = window.foldedContentMap.get(storageKey);
  }

  // If no storage key or data not found, try to parse the line manually
  if (!foldData) {
    const match = foldedLine.match(/^(\s*<[^>]+>)\s*⋯\s*(<\/[^>]+>)/);
    if (!match) {
      return; // Malformed fold
    }
    // Can't unfold without stored content
    return;
  }

  const indentation = foldedLine.substring(0, foldedLine.indexOf(foldData.openingTag));

  // Reconstruct the unfolded text
  const newLines = [
    ...lines.slice(0, foldedLineIndex),
    `${indentation}${foldData.openingTag}`,
    ...foldData.content.split("\n"),
    `${indentation}${foldData.closingTag}`,
    ...lines.slice(foldedLineIndex + 1)
  ];

  // Update textarea
  const cursorPos = outputXml.selectionStart;
  outputXml.value = newLines.join("\n");

  // Restore cursor position
  outputXml.selectionStart = outputXml.selectionEnd = Math.min(cursorPos, outputXml.value.length);

  // Clean up storage
  if (storageKey) {
    window.foldedContentMap.delete(storageKey);
  }

  // Update UI
  updateSyntaxHighlighting();
  updateGutter();
  scheduleValidation();
}

/**
 * Sync gutter scroll with textarea scroll
 */
function syncGutterScroll() {
  if (codeGutter && outputXml) {
    codeGutter.scrollTop = outputXml.scrollTop;
  }
}

// Initialize gutter
if (outputXml && codeGutter) {
  // Update gutter on content changes
  outputXml.addEventListener("input", () => {
    // Clear folded regions when content changes significantly
    // (to avoid stale fold state)
    updateGutter();
  });

  // Sync gutter scroll with textarea scroll
  outputXml.addEventListener("scroll", syncGutterScroll);

  // Initial gutter render
  updateGutter();
}

initTheme();
bootstrap().catch((error) => {
  console.error(error);
  logMessage("error", error.message);
  alert(error.message);
});
