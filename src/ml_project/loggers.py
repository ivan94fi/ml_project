"""This module encloses loggers for the train procedure."""


class Observer:
    """Observer class for observer pattern."""

    def __init__(self, subjects):
        self.subjects = subjects
        for subject in self.subjects:
            subject.register(self)

    def update(self, subject):
        """Abstract method for updates from subjects."""
        raise NotImplementedError

    def add_subject(self, subject):
        """Add another subject."""
        self.subjects.append(subject)
        subject.register(self)


class Subject:
    """Subject class for observer pattern."""

    def __init__(self):
        self.observers = []

    def register(self, observer):
        """Register an observer to this subject's updates."""
        if isinstance(observer, Observer):
            self.observers.append(observer)

    def unregister(self, observer):
        """Unregister the observer from updates."""
        self.observers.remove(observer)

    def notify(self):
        """Notify observers about updates."""
        for observer in self.observers:
            observer.update(self)


class ConditionalLogger(Observer):
    """Configure logging with conditions.

    Parameters
    ----------
    subjects : list of subjects
        The observed subjects
    interval : int
        The interval at which the logs can be emitted
    active_phase : str
        Emit logs only for this phase (the default is None)

    """

    def __init__(self, subjects, interval, active_phase=None):
        super().__init__(list(subjects.values()))

        self.step_subject = subjects.get("step")
        self.phase_subject = subjects.get("phase")

        if active_phase is not None and self.phase_subject is None:
            raise ValueError(
                "When an active phase is specified, there must be a phase subject."
            )

        self.step = None
        self.phase = None
        self.active_phase = active_phase
        self.interval = interval

    def should_log(self):
        """Decide when this logger can emit logs."""
        if self.active_phase is None or self.active_phase == self.phase:
            if self.interval is not None and self.step % self.interval == 0:
                return True
        return False

    def update(self, subject):
        """Handle updates to values from subjects."""
        if subject not in self.subjects:
            raise ValueError("Unkown subject")
        if subject == self.step_subject:
            self.step = subject.next_val
        if subject == self.phase_subject:
            self.phase = subject.next_val


class IterableSubject(Subject):
    """Wrap an iterable in a subject.

    When the iterable is iterated, a notification of its value is emitted to all
    observers.
    """

    def __init__(self):
        super().__init__()
        self.iterable = None
        self.next_val = None

    def __iter__(self):
        """Iterate the inner iterable and emit notifications."""
        self.check_iterable()
        for element in self.iterable:
            self.next_val = element
            self.notify()
            yield element

    def iter(self, iterable):
        """Set the iterable.

        Should be used when defininig loops.
        """
        self.iterable = iterable
        return self

    def check_iterable(self):
        """Make sure that an iterable has been set."""
        if self.iterable is None:
            raise ValueError("Iterable not set.")


class CounterSubject(IterableSubject):
    """Wrap an iterable in a subject.

    When the iterable is iterated, a notification of the current index is
    emitted to all observers.
    """

    def __iter__(self):
        """Iterate the inner iterable and emit notifications."""
        self.check_iterable()
        for index, element in enumerate(self.iterable):
            self.next_val = index
            self.notify()
            yield element
