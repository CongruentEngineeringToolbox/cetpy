import cetpy.Configuration
from cetpy.Configuration import Session

active_session = Session(None)
sessions = []
cetpy.Configuration.refresh_module_dict()


def refresh() -> None:
    """Refresh all cetpy Sessions and the module list."""
    active_session.refresh()
    for sess in sessions:
        sess.refresh()
