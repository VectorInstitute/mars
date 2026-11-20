"""The three released detectors and where their files live."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A released detector on Hugging Face.

    Parameters
    ----------
    id : str
        Short identifier used in the CLI, API and results (``mars``, ``gend``,
        ``dfd-hr``).
    name : str
        Display name.
    repo : str
        Hugging Face repository.
    built_in_attribution : bool
        Whether the graph outputs ``frame_contrib``, ``patch_contrib`` and ``gate``
        (MARS only).
    """

    id: str
    name: str
    repo: str
    built_in_attribution: bool


MODELS: dict[str, ModelSpec] = {
    "mars": ModelSpec(
        id="mars",
        name="MARS",
        repo="vector-institute/mars",
        built_in_attribution=True,
    ),
    "gend": ModelSpec(
        id="gend",
        name="GenD",
        repo="vector-institute/mars-gend",
        built_in_attribution=False,
    ),
    "dfd-hr": ModelSpec(
        id="dfd-hr",
        name="DFD-HR",
        repo="vector-institute/mars-dfd-hr",
        built_in_attribution=False,
    ),
}

#: Order used for scheduling and display: MARS first.
MODEL_ORDER: tuple[str, ...] = ("mars", "gend", "dfd-hr")


def get_spec(model_id: str) -> ModelSpec:
    """Return the spec for a model id.

    Parameters
    ----------
    model_id : str
        One of ``mars``, ``gend``, ``dfd-hr``.

    Returns
    -------
    ModelSpec
        The model's spec.

    Raises
    ------
    ValueError
        If the id is unknown.

    Examples
    --------
    >>> get_spec("mars").name
    'MARS'
    """
    try:
        return MODELS[model_id]
    except KeyError:
        known = ", ".join(MODEL_ORDER)
        raise ValueError(f"unknown model {model_id!r}; choose from {known}") from None


def parse_models(value: str | list[str] | None) -> list[str]:
    """Turn a comma-separated string or list into an ordered list of known model ids.

    Parameters
    ----------
    value : str or list of str or None
        For example ``"mars,gend"``. ``None`` or ``""`` selects MARS only (the default);
        ``"all"`` selects all three.

    Returns
    -------
    list of str
        Model ids in display order, without duplicates.

    Examples
    --------
    >>> parse_models("gend,mars")
    ['mars', 'gend']
    >>> parse_models(None)
    ['mars']
    >>> parse_models("all")
    ['mars', 'gend', 'dfd-hr']
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return [MODEL_ORDER[0]]
    if value == "all":
        return list(MODEL_ORDER)
    items = value.split(",") if isinstance(value, str) else value
    wanted = {get_spec(item.strip().lower()).id for item in items if item.strip()}
    if not wanted:
        raise ValueError("select at least one model")
    return [m for m in MODEL_ORDER if m in wanted]
