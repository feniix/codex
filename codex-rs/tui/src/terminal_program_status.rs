//! Emits Codex lifecycle state through terminal status protocols.
//!
//! OSC 7501 reports lifecycle state and blocking reasons to terminal sessions.
//! Direct iTerm2 sessions also receive OSC 21337 status, including bounded
//! activity text. The process-wide cache is used for exit, panic, and terminal
//! handoff cleanup.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::io::IsTerminal;
use std::io::stdout;
use std::sync::Mutex;
use std::sync::OnceLock;

use codex_terminal_detection::TerminalBackend;
use codex_terminal_detection::TerminalInfo;
use codex_terminal_detection::TerminalName;
use codex_terminal_detection::terminal_info;
use crossterm::Command;
use ratatui::crossterm::execute;

static SESSION_STATUS_STATE: SessionStatusState = SessionStatusState::new();
static TERMINAL_PROGRAM_STATUS_SUPPORT: OnceLock<TerminalProgramStatusSupport> = OnceLock::new();

// VTE 0.54 introduced a parser that safely ignores unknown OSC sequences.
const MIN_VTE_VERSION_WITH_SAFE_OSC: u32 = 5400;

struct SessionStatusState {
    last_emitted: Mutex<BTreeMap<Option<&'static str>, EmittedSessionStatus>>,
}

struct EmittedSessionStatus {
    status: CachedStatus,
    detail: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CachedStatus {
    Reported(ProgramStatus),
    Invalidated,
    PendingClear,
}

impl SessionStatusState {
    const fn new() -> Self {
        Self {
            last_emitted: Mutex::new(BTreeMap::new()),
        }
    }

    fn update(
        &self,
        id: Option<&'static str>,
        status: Option<ProgramStatus>,
        detail: &str,
        mut emit: impl FnMut(Option<&'static str>, Option<ProgramStatus>) -> io::Result<()>,
    ) -> io::Result<()> {
        // Keep the terminal write and cache mutation together so panic-hook cleanup cannot
        // interleave with a draw from another thread.
        let mut last_emitted = self
            .last_emitted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A guard can go away even when its final write fails. The next draw
        // retries that clear, unless this record is itself being updated.
        last_emitted.retain(|&other, entry| {
            other == id || entry.status != CachedStatus::PendingClear || emit(other, None).is_err()
        });
        let previous = last_emitted.get(&id);
        if let Some(status) = status {
            if previous.is_some_and(|entry| {
                entry.status == CachedStatus::Reported(status) && entry.detail == detail
            }) {
                return Ok(());
            }
            emit(id, Some(status))?;
            last_emitted.insert(
                id,
                EmittedSessionStatus {
                    status: CachedStatus::Reported(status),
                    detail: detail.to_owned(),
                },
            );
        } else if let Some(entry) = last_emitted.get_mut(&id) {
            entry.status = CachedStatus::PendingClear;
            emit(id, None)?;
            last_emitted.remove(&id);
        }
        Ok(())
    }

    // Clear every live child. On handoff keep the root dirty for the next draw.
    // Retain failed clears for a later retry, and still attempt the other records.
    fn cleanup(
        &self,
        invalidate_root: bool,
        mut clear: impl FnMut(Option<&'static str>) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut first_error = Ok(());
        self.last_emitted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|&id, entry| {
                if invalidate_root && id.is_none() {
                    entry.status = CachedStatus::Invalidated;
                    return true;
                }
                if let Err(err) = clear(id) {
                    entry.status = CachedStatus::PendingClear;
                    if first_error.is_ok() {
                        first_error = Err(err);
                    }
                    return true;
                }
                false
            });
        first_error
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TerminalProgramStatusSupport {
    osc_7501: bool,
    osc_21337: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgramStatus {
    Idle,
    Working,
    Blocked(ProgramStatusKind),
    Done,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgramStatusKind {
    Permission,
    Question,
    Auth,
}

impl ProgramStatus {
    fn iterm_label(self) -> &'static str {
        match self {
            Self::Idle | Self::Done | Self::Error => "Idle",
            Self::Working => "Working",
            Self::Blocked(_) => "Waiting",
        }
    }

    fn iterm_indicator(self) -> &'static str {
        match self {
            Self::Idle | Self::Done | Self::Error => "#00d75f",
            Self::Working => "#ff9500",
            Self::Blocked(_) => "#5f87ff",
        }
    }

    fn protocol_state(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Working => "working",
            Self::Blocked(_) => "blocked",
            Self::Done => "done",
            Self::Error => "error",
        }
    }
}

pub(crate) fn set_terminal_program_status(
    status: ProgramStatus,
    detail: Option<&str>,
) -> io::Result<()> {
    set_record(None, status, detail)
}

fn set_record(
    id: Option<&'static str>,
    status: ProgramStatus,
    detail: Option<&str>,
) -> io::Result<()> {
    if cfg!(test) {
        return Ok(());
    }

    let support = current_terminal_program_status_support();
    if !support.osc_7501 {
        return Ok(());
    }

    let detail = if id.is_none() && support.osc_21337 {
        sanitize_iterm_session_detail(detail.unwrap_or_default())
    } else {
        String::new()
    };
    SESSION_STATUS_STATE.update(id, Some(status), &detail, |id, status| match status {
        Some(status) => execute!(
            stdout(),
            SetTerminalProgramStatus {
                id,
                status,
                iterm_detail: &detail,
                emit_iterm_status: support.osc_21337,
            }
        ),
        None => clear_record(id),
    })
}

pub(crate) fn clear_terminal_program_status() -> io::Result<()> {
    SESSION_STATUS_STATE.cleanup(false, clear_record)
}

fn clear_record(id: Option<&'static str>) -> io::Result<()> {
    execute!(
        stdout(),
        ClearTerminalProgramStatus {
            id,
            emit_iterm_status: current_terminal_program_status_support().osc_21337,
        }
    )
}

/// Clear child records and force the next active draw to re-emit the root status.
pub(crate) fn invalidate_terminal_program_status() {
    if let Err(err) = SESSION_STATUS_STATE.cleanup(true, clear_record) {
        tracing::debug!(%err, "failed to clear terminal program status after handoff");
    }
}

/// Owns the OSC record for the currently visible startup dialog.
#[derive(Default)]
pub(crate) struct DialogStatusGuard {
    id: Option<&'static str>,
}

impl DialogStatusGuard {
    pub(crate) fn report(&mut self, id: &'static str, kind: ProgramStatusKind) {
        if self.id != Some(id) {
            self.clear();
            self.id = Some(id);
        }
        if let Err(err) = set_record(self.id, ProgramStatus::Blocked(kind), None) {
            tracing::debug!(%err, id, "failed to set dialog status");
        }
    }

    pub(crate) fn clear(&mut self) {
        if let Some(id) = self.id.take()
            && let Err(err) =
                SESSION_STATUS_STATE.update(Some(id), None, "", |id, _| clear_record(id))
        {
            tracing::debug!(%err, id, "failed to clear dialog status");
        }
    }
}

impl Drop for DialogStatusGuard {
    fn drop(&mut self) {
        self.clear();
    }
}

fn sanitize_iterm_session_detail(detail: &str) -> String {
    let detail = crate::terminal_title::sanitize_terminal_title(
        detail,
        crate::terminal_title::TitleEncoding::Unicode,
    );
    let mut escaped = String::with_capacity(detail.len());
    for ch in detail.chars() {
        if matches!(ch, ';' | '\\') {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn current_terminal_program_status_support() -> TerminalProgramStatusSupport {
    *TERMINAL_PROGRAM_STATUS_SUPPORT.get_or_init(|| {
        terminal_program_status_support(
            &terminal_info(),
            /*stdout_is_terminal*/ stdout().is_terminal(),
            /*screen_active*/ std::env::var_os("STY").is_some(),
        )
    })
}

fn terminal_program_status_support(
    terminal: &TerminalInfo,
    stdout_is_terminal: bool,
    screen_active: bool,
) -> TerminalProgramStatusSupport {
    let osc_7501 = stdout_is_terminal
        && match terminal.backend {
            Some(TerminalBackend::Vte { version }) => {
                version.is_none_or(|version| version >= MIN_VTE_VERSION_WITH_SAFE_OSC)
            }
            None => true,
        };
    let osc_21337 = osc_7501
        && terminal.name == TerminalName::Iterm2
        && terminal.multiplexer.is_none()
        && !screen_active;
    TerminalProgramStatusSupport {
        osc_7501,
        osc_21337,
    }
}

#[derive(Clone, Copy, Debug)]
struct SetTerminalProgramStatus<'a> {
    id: Option<&'a str>,
    status: ProgramStatus,
    iterm_detail: &'a str,
    emit_iterm_status: bool,
}

impl Command for SetTerminalProgramStatus<'_> {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        if self.id.is_none() && self.emit_iterm_status {
            write!(
                f,
                "\x1b]21337;status={};indicator={};status-color=;detail={}\x07",
                self.status.iterm_label(),
                self.status.iterm_indicator(),
                self.iterm_detail,
            )?;
        }
        write!(f, "\x1b]7501;state={}", self.status.protocol_state())?;
        if self.id.is_none() {
            write!(f, ":app=codex")?;
        }
        if let ProgramStatus::Blocked(kind) = self.status {
            let kind = match kind {
                ProgramStatusKind::Permission => "permission",
                ProgramStatusKind::Question => "question",
                ProgramStatusKind::Auth => "auth",
            };
            write!(f, ":kind={kind}")?;
        }
        if let Some(id) = self.id {
            write!(f, ":id={id}:app=codex")?;
        }
        write!(f, "\x1b\\")
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> io::Result<()> {
        Err(io::Error::other(
            "tried to set terminal program status using WinAPI; use ANSI instead",
        ))
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug)]
struct ClearTerminalProgramStatus<'a> {
    id: Option<&'a str>,
    emit_iterm_status: bool,
}

impl Command for ClearTerminalProgramStatus<'_> {
    fn write_ansi(&self, f: &mut impl fmt::Write) -> fmt::Result {
        if self.id.is_none() && self.emit_iterm_status {
            write!(f, "\x1b]21337;status=;indicator=;status-color=;detail=\x07")?;
        }
        write!(f, "\x1b]7501;state=clear")?;
        if let Some(id) = self.id {
            write!(f, ":id={id}")?;
        }
        write!(f, "\x1b\\")
    }

    #[cfg(windows)]
    fn execute_winapi(&self) -> io::Result<()> {
        Err(io::Error::other(
            "tried to clear terminal program status using WinAPI; use ANSI instead",
        ))
    }

    #[cfg(windows)]
    fn is_ansi_code_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "iterm_session_status_tests.rs"]
mod tests;
