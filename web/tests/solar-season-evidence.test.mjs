import test from 'node:test';
import assert from 'node:assert/strict';
import { readSolarEvidence, solarChartRows } from '../src/solar-season-evidence.ts';

// Synthetic parser fixture, never served or exported as research results.
function fixture() {
  const roles = ['original_development', 'new_development', 'original_reference', 'reserved_new_geography'];
  const row = (test_mode, geography_role, missing = false) => ({ test_mode, geography_role, paired: missing ? 0 : 2, dates: missing ? 0 : 1,
    baseline_mae: missing ? null : 4, candidate_mae: missing ? null : 3, baseline_balanced_mae: missing ? null : 4, candidate_balanced_mae: missing ? null : 3,
    baseline_bias: missing ? null : 1, candidate_bias: missing ? null : -1, baseline_above_5_fraction: missing ? null : .2,
    candidate_above_5_fraction: missing ? null : .1, baseline_above_7_fraction: missing ? null : .1,
    candidate_above_7_fraction: missing ? null : .05, delta_balanced_mae: missing ? null : -1, outcome: missing ? 'missing' : 'improved' });
  return { status: 'complete', generated_at_utc: '2026-09-19T00:00:00Z', baseline_label: 'Expanded baseline', candidate_label: 'Solar-season candidate',
    description: 'Synthetic testing only', target_mae_c: 3, qualified: false,
    comparisons: ['geography', 'month', 'reference'].flatMap(mode => roles.map(role => row(mode, role, role === 'reserved_new_geography' || (mode === 'reference') !== (role === 'original_reference')))),
    regions: Array.from({ length: 48 }, (_, i) => ['day', 'night'].map(phase => {
      const role = i === 0 ? 'original_reference' : i < 12 ? 'original_development' : i < 40 ? 'new_development' : 'reserved_new_geography';
      return { ...row(i === 0 ? 'reference' : 'geography', role, i >= 40), region_id: i === 0 ? 'cabauw' : `site${i}`, phase };
    })).flat(), download_url: '/research/solar-season-region-phase.csv', cases_download_url: '/research/solar-season-date-phase.csv',
    source: { audit_sha256: 'a'.repeat(64), metrics_sha256: 'b'.repeat(64), trial_sha256: 'c'.repeat(64) } };
}
const read = d => readSolarEvidence(d, 'https://example.test');
test('five chart populations remain separate; all 96 outcomes and reserved rows remain', () => {
  const data = read(fixture());
  assert.equal(data.regions.length, 96); assert.equal(solarChartRows(data.comparisons).length, 5);
  assert.equal(data.regions.filter(r => r.paired === 0).length, 16);
  assert.ok(solarChartRows(data.comparisons).every(r => r.geography_role !== 'reserved_new_geography'));
});
test('missing errors, incomplete rows, duplicate identities and absent proof are rejected', () => {
  const cases = [d => { d.regions[95].candidate_mae = 0; }, d => { d.regions.pop(); },
    d => { d.regions[1] = d.regions[0]; }, d => { d.source.audit_sha256 = ''; },
    d => { d.regions[0].candidate_balanced_mae = null; }, d => { d.status = 'evaluating'; }];
  for (const mutate of cases) { const d = fixture(); mutate(d); assert.throws(() => read(d)); }
});
test('only explicit candidate names and safe same-origin research CSV paths are accepted', () => {
  for (const path of ['https://elsewhere.test/a.csv', '/research/../secrets.csv', '/research/a.csv?q=1', '//elsewhere.test/a.csv']) {
    const d = fixture(); d.download_url = path; assert.throws(() => read(d));
  }
  const d = fixture(); d.baseline_label = 'Original data'; assert.throws(() => read(d));
});
