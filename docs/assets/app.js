'use strict';

const capacities = [0.05, 0.10, 0.15, 0.20, 0.30];
const capacityInput = document.getElementById('capacity');
const number = new Intl.NumberFormat('en-US');
const percent = (value) => `${(value * 100).toFixed(1)}%`;
let projectData;

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function renderCapacity() {
  const fraction = capacities[Number(capacityInput.value)];
  const rows = projectData.capacity.filter((row) => row.capacity === fraction);
  const selected = rows.find((row) => row.policy === 'Selected model');
  const admin = rows.find((row) => row.policy === 'Administrative logistic');
  const prior = rows.find((row) => row.policy === 'Lowest previous score first');
  const testCount = projectData.summary.split_records['Final test'];
  const failureCount = selected.failures_in_queue + selected.failures_outside_queue;
  const randomExpected = selected.slots * failureCount / testCount;
  const fractionLabel = `${Math.round(fraction * 100)}%`;

  capacityInput.setAttribute('aria-valuetext', `${Math.round(fraction * 100)} percent capacity`);
  setText('capacity-percent', fractionLabel);
  setText('slot-count', `${number.format(selected.slots)} of ${number.format(testCount)} students`);
  setText('failures-reached', number.format(selected.failures_in_queue));
  setText('extra-reached', `${number.format(selected.failures_in_queue - admin.failures_in_queue)} more than the administrative baseline`);
  setText('queue-precision', percent(selected.precision_at_capacity));
  setText('failure-recall', percent(selected.recall_at_capacity));
  setText('failures-outside', number.format(selected.failures_outside_queue));

  const policies = [
    {label: 'Student Support Queue', value: selected.failures_in_queue, style: 'model'},
    {label: 'Administrative baseline', value: admin.failures_in_queue, style: 'admin'},
    {label: 'Lowest previous score first', value: prior.failures_in_queue, style: 'prior'},
    {label: 'Random allocation (expected)', value: randomExpected, style: 'random'}
  ];
  const bars = document.getElementById('comparison-bars');
  bars.replaceChildren();
  for (const policy of policies) {
    const row = document.createElement('div');
    row.className = `compare-row ${policy.style}`;
    const label = document.createElement('div');
    label.className = 'compare-label';
    const name = document.createElement('span');
    name.textContent = policy.label;
    const value = document.createElement('strong');
    value.textContent = number.format(Math.round(policy.value * 10) / 10);
    label.append(name, value);
    const track = document.createElement('div');
    track.className = 'compare-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div');
    fill.className = 'compare-fill';
    fill.style.width = `${policy.value / selected.slots * 100}%`;
    track.append(fill);
    row.append(label, track);
    bars.append(row);
  }
}

async function initialize() {
  try {
    capacityInput.disabled = true;
    const response = await fetch('data/project.json');
    if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
    projectData = await response.json();
    if (!projectData.summary || !Array.isArray(projectData.capacity)) throw new Error('Missing project evidence.');
    renderCapacity();
    capacityInput.addEventListener('input', renderCapacity);
    capacityInput.disabled = false;
  } catch (error) {
    setText('data-status', 'The interactive evidence could not load. Read the notebook for the complete results.');
    console.error(error);
  }
}

initialize();
