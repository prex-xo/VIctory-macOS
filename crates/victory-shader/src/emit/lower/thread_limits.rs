//! A dispatch counted in threads (the console's `DispatchX` with its
//! thread-dimensions flag): the lanes past the count leave at the start
//! ([`Lower::leave_past_the_count`]), in the programs that can.

use super::*;

impl Lower<'_> {
    /// Whether the lanes past a dispatch's count of threads can leave at
    /// the start ([`Lower::leave_past_the_count`]): in a compute program
    /// of one dimension whose lanes never wait for each other at a group
    /// barrier, the console's or one the host's narrow waves need, nor
    /// pass values between 32-lane halves (barriers want every lane), nor
    /// run as literal halves (which need every lane of the group to take
    /// the same path).
    pub(super) fn can_leave_past_a_count(&self) -> bool {
        let StageIo::Cs(cs) = self.x.native.io else {
            return false;
        };
        let barrier = self
            .x
            .p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "s_barrier");
        cs.threads[0] > 1
            && cs.threads[1..] == [1, 1]
            && !barrier
            && !self.x.st.literal
            && self.split_barriers.is_empty()
            && (!self.halves || self.one_host_wave())
    }

    /// A dispatch whose counts are threads (the console's `DispatchX` with
    /// its thread-dimensions flag) may end in a part of a group, whose other
    /// lanes the console never starts: the program counts the lanes of a
    /// wave and stores by thread. Here the host runs whole groups and says
    /// in [`super::super::USER_DISPATCH_THREADS_X`] how many threads along x run
    /// (0: all); the lanes past that leave before anything else, so that
    /// wave counts, first-lane reads and stores know only those that run.
    /// After DexNom/VIctory bc3cd4b0, which did so for two programs by
    /// their digests.
    pub(super) fn leave_past_the_count(&mut self) {
        let original = self.s.entry;
        let width = self.s.threads.expect("compute thread count")[0];
        // A dword past the console's user data.
        let layout = self
            .x
            .native
            .layout
            .iter()
            .map(|e| u32::from(e.user_data) + e.sgprs);
        let index = match self.x.native.io {
            StageIo::Cs(cs) => layout
                .chain([u32::from(cs.user_sgprs())])
                .max()
                .unwrap_or(0),
            _ => unreachable!("compute"),
        };
        let limit = self.user_at(UserData {
            index,
            slot: 0,
            root: RootType::Other(super::super::USER_DISPATCH_THREADS_X),
            dword: width,
        });
        let Val::I32(group) = self.compute_id("groupId.i32", 94, 0) else {
            unreachable!()
        };
        let Val::I32(thread) = self.thread_id(0) else {
            unreachable!()
        };
        let width = self.s.m.i32(width);
        let first = self.s.m.bin(original, Bin::Mul, group, width);
        let x = self.s.m.bin(original, Bin::Add, first, thread);
        let zero = self.s.m.i32(0);
        let all = self.s.m.cmp(original, pred::EQ, limit, zero);
        let inside = self.s.m.cmp(original, pred::IULT, x, limit);
        let runs = self.s.m.bin(original, Bin::Or, all, inside);
        let (body, done) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(original, Term::CondBr(runs, body, done));
        self.s.m.terminate(done, Term::Ret);
        self.s.entry = body;
    }
}
