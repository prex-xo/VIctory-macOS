//! A branch both of whose targets are loop back edges, each under the loop
//! iteration limit ([`LOOP_LIMIT`]).

use super::*;

impl Lower<'_> {
    /// The terminator of IR block `b`'s branch on `cond` whose both targets,
    /// `taken` and `not_taken`, are loop headers it jumps back to (two
    /// nested loops whose latches share a block, say). A single back edge
    /// leaves its loop at the limit by the branch's other side, a forward
    /// one; here both sides go back, so each gets a block of its own that
    /// counts the iteration of its loop and, at the limit, leaves the
    /// program as an unconditional back edge does ([`Lower::abort_block`]).
    /// Below the limit both go where the console goes. In whole-wave control
    /// flow every lane of the console wave takes the same side, with the
    /// same counts, so the halves reach the limit together.
    pub(super) fn two_back_edges(
        &mut self,
        b: usize,
        cond: V,
        taken: usize,
        not_taken: usize,
    ) -> Term {
        let side = |l: &mut Self, to: usize| {
            let c = l.cur;
            let (phi, _) = l.counters[&to];
            let one = l.s.m.i32(1);
            let next = l.s.m.bin(c, Bin::Add, phi, one);
            let limit = l.s.m.i32(super::super::loop_limit());
            let ok = l.s.m.cmp(c, pred::IULT, next, limit);
            let check = l.s.m.new_block();
            l.counters
                .get_mut(&to)
                .expect("header")
                .1
                .push((next, check));
            // The header's phis take this edge's values from `check`.
            l.edges.insert((b, to), check);
            let abort = l.abort_block();
            let header = l.block[to].expect("reachable");
            l.s.m.terminate(check, Term::CondBr(ok, header, abort));
            check
        };
        let t = side(self, taken);
        let f = side(self, not_taken);
        Term::CondBr(cond, t, f)
    }
}
