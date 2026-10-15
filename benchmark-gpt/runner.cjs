// Each job runs in a separate VM with a one-second execution limit.
const fs = require('node:fs');
const vm = require('node:vm');
const jobs = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = jobs.map(job => {
  try {
    if (job.literal) {
      vm.runInNewContext(`(${job.literal})`, {}, {timeout: 1000});
    }
    return vm.runInNewContext(`
      const r = new RegExp(job.pattern, job.flags);
      ({bad_match: job.match.filter(s => !r.test(s)),
        bad_no_match: job.no_match.filter(s => r.test(s))})
    `, {job}, {timeout: 1000});
  } catch (error) {
    return {error: String(error)};
  }
});
process.stdout.write(JSON.stringify(results));
