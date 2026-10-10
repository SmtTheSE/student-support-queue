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
  const simpler = rows.find((row) => row.policy === 'Administrative logistic');
  const testCount = projectData.summary.split_records['Final test'];
  const extra = selected.failures_in_queue - simpler.failures_in_queue;
  const places = number.format(selected.slots);
  const fractionLabel = `${Math.round(fraction * 100)}%`;

  capacityInput.setAttribute('aria-valuetext', `${Math.round(fraction * 100)} percent: ${places} places`);
  setText('capacity-percent', fractionLabel);
  setText('slot-count', `${places} places for ${number.format(testCount)} students`);
  setText('failures-reached', number.format(selected.failures_in_queue));
  setText('baseline-count', number.format(simpler.failures_in_queue));
  setText('extra-count', `${extra >= 0 ? '+' : ''}${number.format(extra)}`);
  setText('result-story', `This tool included ${number.format(extra)} more students who later did not pass, using the same ${places} places.`);
  setText('failures-outside', number.format(selected.failures_outside_queue));
  setText('queue-precision', percent(selected.precision_at_capacity));
  setText('failure-recall', percent(selected.recall_at_capacity));
}

async function initialize() {
  try {
    capacityInput.disabled = true;
    const response = await fetch('data/project.json');
    if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
    projectData = await response.json();
    const testCount = projectData.summary?.split_records?.['Final test'];
    if (!Number.isFinite(testCount) || testCount <= 0 || !Array.isArray(projectData.capacity)) {
      throw new Error('Missing project results.');
    }
    // Every slider position must have a complete, comparable pair of results.
    for (const fraction of capacities) {
      const rows = projectData.capacity.filter((row) => row.capacity === fraction);
      const selected = rows.find((row) => row.policy === 'Selected model');
      const simpler = rows.find((row) => row.policy === 'Administrative logistic');
      if (!selected || !simpler || selected.slots !== simpler.slots ||
          ![selected.slots, selected.failures_in_queue, simpler.failures_in_queue,
            selected.failures_outside_queue, selected.precision_at_capacity,
            selected.recall_at_capacity].every(Number.isFinite)) {
        throw new Error('Incomplete list comparison.');
      }
    }
    renderCapacity();
    capacityInput.addEventListener('input', renderCapacity);
    capacityInput.disabled = false;
  } catch (error) {
    setText('data-status', 'The slider could not load. The 10% example is shown above. Open the notebook to see all results.');
    console.error(error);
  }
}

initialize();
