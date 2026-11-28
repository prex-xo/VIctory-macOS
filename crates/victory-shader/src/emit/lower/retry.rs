//! The lowering's entry: a program in the per-lane form, and a 64-thread
//! console wave64 on a 32-lane host that form refuses at a cross-half gate
//! again with whole-wave control flow (#435).

use super::*;

#[cfg(any(test, all(windows, feature = "host")))]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_pruned(parts, None, false, false)
}

#[cfg(any(test, all(windows, feature = "host")))]
pub(crate) fn lower_pruned(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    literal_halves: bool,
) -> Result<Lowered, ShaderError> {
    lower_limited(parts, keep, per_vertex_center, literal_halves, false)
}

/// [`lower_pruned`]; `limits`: a dispatch counted in threads may end in a
/// part of a group, whose other lanes the console does not start. A
/// compute program that can leave them at the start does
/// ([`Lower::leave_past_the_count`]; never in literal halves). One that
/// cannot (they would miss its group barriers and its exchanges between
/// 32-lane halves) keeps them, outside EXEC from the entry on, as the
/// console has them in its last wave
/// ([`crate::structure::structure_limited`]). Where that form is refused
/// too, the program is as without `limits`: its last group runs whole.
pub(crate) fn lower_limited(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    literal_halves: bool,
    limits: bool,
) -> Result<Lowered, ShaderError> {
    let first = lower_limited_as_structured(parts, keep, per_vertex_center, literal_halves, limits);
    let counted = |l: &Lowered| {
        l.user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::super::USER_DISPATCH_THREADS_X))
    };
    let one_row = matches!(
        &parts.native.io,
        StageIo::Cs(cs) if cs.threads[0] > 1 && cs.threads[1..] == [1, 1]
    );
    if !limits || !one_row || parts.st.limited || first.as_ref().is_ok_and(counted) {
        return first;
    }
    let limited = crate::structure::structure_limited(parts.p).and_then(|st| {
        let native = parts.native;
        let res = crate::resource::track(parts.p, &native.layout, native.stage, Some(&st))?;
        let parts = Parts {
            st: &st,
            res: &res,
            ..parts.clone()
        };
        lower_limited_as_structured(&parts, keep, per_vertex_center, false, limits)
    });
    match limited {
        Ok(limited) if counted(&limited) => Ok(limited),
        _ => first,
    }
}

/// [`lower_limited`] with the lanes as `parts.st` has them.
fn lower_limited_as_structured(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    literal_halves: bool,
    limits: bool,
) -> Result<Lowered, ShaderError> {
    // Retry refusals from the legacy cross-half gates with whole-wave control flow.
    let original = lower_with_structure(parts, keep, per_vertex_center, limits);
    if !literal_halves
        || !parts.host_wave32
        || !parts.native.wave64
        || !matches!(&parts.native.io, StageIo::Cs(cs) if cs.threads.iter().product::<u32>() == 64)
        || !matches!(
            &original,
            Err(ShaderError::Unsupported {
                what: halves::MASK_ACROSS
                    | LANE_MASK
                    | MASK_CYCLE
                    | MASK_ELSEWHERE
                    | halves::LDS_ACROSS
                    | halves::REDUCTION_ACROSS
                    | halves::FIRST_ACROSS
                    | halves::READ_ACROSS
                    | halves::WRITE_ACROSS,
                ..
            })
        )
    {
        return original;
    }
    lower_literal(parts, keep, per_vertex_center)
}

/// The whole-wave translation of a 64-thread wave64 compute program on a
/// 32-lane host, tried when the per-lane halves translation refuses it.
pub(crate) fn lower_literal(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
) -> Result<Lowered, ShaderError> {
    let native = parts.native;
    // Register-relative move bounds and resource paths must use the same
    // control flow as lowering. The per-lane move bounds do not apply here.
    let p = crate::ir::build(&native.patched_code(), true, native.vgprs)?;
    let st = crate::structure::split_literal(&p)?;
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
    if !res.waterfall.reads.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a resource waterfall in literal wave64 halves",
        });
    }
    let io = crate::io::raise(&p, native)?;
    lower_with_structure(
        &Parts {
            p: &p,
            st: &st,
            res: &res,
            io: &io,
            ..parts.clone()
        },
        keep,
        per_vertex_center,
        // A group in literal halves takes one path: its lanes never leave
        // past a count of threads (`Lower::can_leave_past_a_count`).
        false,
    )
}

fn lower_with_structure(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    limits: bool,
) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    // Divergent sites where the halves meet (`lower::meet`): a pass that
    // finds new ones goes again with them laid out.
    let mut plan: Set<MeetSite> = Set::default();
    loop {
        let found = std::cell::RefCell::new(Set::default());
        let r = Lower::new(parts, &bad, keep, per_vertex_center, &plan, &found).run(limits);
        let found = found.into_inner();
        if !found.is_empty() {
            plan.extend(found);
            continue;
        }
        match r {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}
