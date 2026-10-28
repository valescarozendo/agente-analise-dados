from types import SimpleNamespace
import pytest
from analista_dados.reporter import InsightReporter, ReportError, build_user_message


class FakeClient:
    def __init__(self, text="## 1. Sumário executivo\nOK", stop_reason="end_turn"):
        self.calls = []
        self._response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_user_message_includes_context_and_profile():
    message = build_user_message({"overview": {"rows": 3}}, context="Varejo 2024")
    assert "Varejo 2024" in message
    assert '"rows":3' in message


def test_generate_sends_profile_and_returns_text():
    client = FakeClient()
    reporter = InsightReporter(model="modelo-teste", client=client)
    report = reporter.generate({"overview": {"rows": 3}})
    assert report.startswith("## 1.")


def test_empty_response_raises_report_error():
    reporter = InsightReporter(client=FakeClient(text=""))
    with pytest.raises(ReportError):
        reporter.generate({})