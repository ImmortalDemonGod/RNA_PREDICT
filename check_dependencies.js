import fs from 'fs';

// Read the tasks.json file
const tasksData = JSON.parse(fs.readFileSync('./tasks/tasks.json', 'utf8'));
const tasks = tasksData.tasks;

// Function to check if a task is blocked
function isTaskBlocked(task) {
  if (!task.dependencies || task.dependencies.length === 0) {
    return false;
  }

  for (const depId of task.dependencies) {
    const depTask = tasks.find(t => t.id.toString() === depId.toString());
    if (!depTask || depTask.status !== 'done') {
      return true;
    }
  }

  return false;
}

// Print all tasks with their status and dependency information
console.log('ID | Title | Status | Dependencies | Blocked');
console.log('---|-------|--------|--------------|--------');

for (const task of tasks) {
  const deps = task.dependencies ? task.dependencies.join(', ') : 'None';
  const blocked = isTaskBlocked(task) ? 'Yes' : 'No';
  console.log(`${task.id} | ${task.title} | ${task.status} | ${deps} | ${blocked}`);
}

// Print the next unblocked tasks
console.log('\nNext unblocked tasks:');
const unblocked = tasks.filter(task => task.status === 'pending' && !isTaskBlocked(task));
for (const task of unblocked) {
  console.log(`${task.id} - ${task.title}`);
}
