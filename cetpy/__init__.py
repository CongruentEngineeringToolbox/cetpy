import cetpy.Configuration
from cetpy.Configuration import Session

sessions = []  # Has to happen before Session initialisation
active_session = Session(None)
# cetpy.Configuration.refresh_module_dict()  # Turned off until further notice, to avoid automatic tensorflow import
# and issues with virtual environments until the overlap function is improved.


def refresh() -> None:
    """Refresh all cetpy Sessions and the module list."""
    active_session.refresh()
    for sess in sessions:
        sess.refresh()


def new_session(directory: str | None, logging_level: str = 'info'):
    """Initialise a new working session."""
    Session(directory, logging_level)

