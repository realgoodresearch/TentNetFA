"""Tests for displacement_tracker/g2_validate_geojson.py."""

from displacement_tracker.g2_validate_geojson import LazyReferenceTypeChoice

# ==========================================================
# --reference-type: which source types the CLI offers
# ==========================================================


def test_reference_type_reads_the_registry_at_parse_time(source_types_registry):
    # Given: validate-geojson's --reference-type parameter type, and a source
    #        type registered AFTER that object was constructed. That is the
    #        production ordering for manual_eval: it is registered on demand
    #        by util/reference_data.py, long after g2's decorators have run
    #        at import time.
    choice = LazyReferenceTypeChoice()
    source_types_registry["late_arrival"] = (object, frozenset())

    # When: Click reads the permitted values, as it does on every parse
    choices = choice.choices

    # Then: the late registration is offered, so the list is being read now
    #       rather than frozen when the decorator ran — and manual_eval comes
    #       with it. A click.Choice over a list captured at decoration time
    #       rejects `--reference-type manual_eval` however the registry looks
    #       by the time the command runs.
    assert "late_arrival" in choices
    assert "manual_eval" in choices
    assert choices == ("late_arrival", "manual_eval", "raster", "unosat", "vector")
