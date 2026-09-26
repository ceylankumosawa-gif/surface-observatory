from collections import Counter
from lst_pilot.coarse_recovery import balanced_select


def candidates():
    out=[];serial=0
    for region in ['greater_london','sioux_falls']:
        for year in [2021,2022,2023]:
            for quarter in range(4):
                for product in ['MOD21','VNP21']:
                    for rank in [1,2]:
                        serial+=1;day=quarter*80+10+rank
                        version='061' if product=='MOD21' else '002'
                        # Distinct physical acquisition time for the two pilots.
                        hm='1100' if region=='greater_london' else '1700'
                        stem=f'{product}.A{year}{day:03d}.{hm}.{version}.2024001000000'
                        out.append({'target':{'region_id':region,'datetime_utc':f'{year}-{quarter*3+2:02d}-20T12:00Z'},
                            'candidate_rank':rank,'record':{'product':product,'stem':stem}})
    return out


def test_recovery_balances_four_outer_groups_and_caps_new_sources():
    selected=balanced_select(candidates(),40)
    assert len(selected)==40
    groups=Counter((x['target']['region_id'],'fit' if x['target']['datetime_utc'][:4]<'2023' else 'eval') for x in selected)
    assert set(groups.values())=={10}
    assert len({x['record']['stem'] for x in selected})==40
    per_target=Counter((x['target']['region_id'],x['target']['datetime_utc'],x['record']['product']) for x in selected)
    assert max(per_target.values())<=2
    for group in groups:
        months={int(x['target']['datetime_utc'][5:7]) for x in selected if (x['target']['region_id'],'fit' if x['target']['datetime_utc'][:4]<'2023' else 'eval')==group}
        assert len(months)==4
    for region in ['greater_london','sioux_falls']:
        years=Counter(x['target']['datetime_utc'][:4] for x in selected if x['target']['region_id']==region)
        assert years=={'2021':5,'2022':5,'2023':10}


def test_recovery_is_deterministic_and_physical_duplicates_do_not_expand_count():
    original=candidates();a=balanced_select(original);b=balanced_select(list(reversed(original)))
    assert a==b
    assert balanced_select(original+original)==a
