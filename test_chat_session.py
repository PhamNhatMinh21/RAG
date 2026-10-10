import contextlib
import io
import os
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from chat import (
    BASE_DIR, create_conversation_chain, format_answer_citations,
    format_documents, normalize_chat_input, run_terminal_chat,
)


class ConversationTests(unittest.TestCase):
    def test_long_history_only_trims_model_context(self):
        history = []
        for index in range(50):
            history.extend([
                {"role": "user", "content": f"Câu {index} " + "văn bản " * 500},
                {"role": "assistant", "content": f"Đáp {index} " + "trả lời " * 500},
            ])
        inputs = normalize_chat_input({"question": "Hỏi tiếp", "history": history})
        self.assertEqual(len(history), 100)
        self.assertLess(len(inputs["history"]), len(history))
        self.assertEqual(inputs["history"][0].type, "human")
        self.assertEqual(inputs["history"][-1].content, history[-1]["content"])

    def test_followup_retrieval_and_prompt_use_history(self):
        prompts, queries = [], []

        def respond(prompt):
            messages = prompt.to_messages()
            prompts.append(messages)
            if "viết lại" in messages[0].content:
                return AIMessage(content="Tỷ lệ thắng mục tiêu trong bài AI điều chỉnh độ khó?")
            return AIMessage(content="Tỷ lệ thắng mục tiêu là 60%.")

        def retrieve(query):
            queries.append(query)
            return [Document(page_content="Mục tiêu 60%", metadata={"source": "paper.pdf", "page_number": 2})]

        chain = create_conversation_chain(RunnableLambda(retrieve), RunnableLambda(respond))
        first = "Bài AI điều chỉnh độ khó nói gì?"
        answer = chain.invoke(first)
        history = [{"role": "user", "content": first}, {"role": "assistant", "content": answer}]
        chain.invoke({"question": "Tỷ lệ đó là bao nhiêu?", "history": history})
        self.assertEqual(queries, [first, "Tỷ lệ thắng mục tiêu trong bài AI điều chỉnh độ khó?"])
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[-1][1].content, first)
        self.assertEqual(prompts[-1][2].content, answer)
        self.assertIn("paper.pdf:2", prompts[-1][0].content)
        self.assertEqual(len(history), 2)
        # Một phiên mới không nhận lịch sử của phiên trước.
        chain.invoke("Câu hỏi mới")
        self.assertEqual(len(prompts[-1]), 2)

    def test_terminal_keeps_running_after_answers_and_error(self):
        chain = Mock()
        chain.invoke.side_effect = ["Đáp án 1", RuntimeError("thử lỗi"), "Đáp án 2"]
        with patch("builtins.input", side_effect=["", "Câu 1", "Lỗi", "Câu 2", "/exit"]):
            with contextlib.redirect_stdout(io.StringIO()):
                run_terminal_chat(chain)
        self.assertEqual(chain.invoke.call_count, 3)
        self.assertEqual(len(chain.invoke.call_args_list[-1].args[0]["history"]), 2)


class CitationTests(unittest.TestCase):
    def test_moves_legacy_citations_to_footer_without_changing_equations(self):
        source = r"C:\Users\phamm\Downloads\test\papers\paper.pdf"
        answer = (
            f"Mục tiêu 60% {{{source}:1}}.\n"
            f"Công thức $p=1/{{1+x}}$ {{{source}:2}}.\n"
            f"Cập nhật độ khó {{{source}; trang: 2}}."
        )
        formatted = format_answer_citations(answer)
        body, footer = formatted.split("**Nguồn đã sử dụng**")
        self.assertNotIn(".pdf", body)
        self.assertNotIn("C:\\", formatted)
        self.assertIn("$p=1/{1+x}$", body)
        self.assertEqual(footer.count("paper.pdf:2"), 1)
        self.assertIn("paper.pdf:1", footer)
        self.assertEqual(format_answer_citations(formatted), formatted)

    def test_only_used_valid_sources_appear(self):
        docs = [
            Document(page_content="Trang một", metadata={"source": r"C:\papers\paper.pdf", "page": 0}),
            Document(page_content="Trang hai", metadata={"source": r"C:\papers\paper.pdf", "page": 1}),
            Document(page_content="Không sử dụng", metadata={"source": "unused.pdf", "page": 2}),
        ]
        context = format_documents(docs)
        self.assertNotIn("C:\\", context)
        self.assertIn("paper.pdf:1", context)
        llm = RunnableLambda(lambda prompt: AIMessage(content=(
            "Ý một [[SOURCE:paper.pdf:2]]. Ý hai [[SOURCE:paper.pdf:2]]. "
            "Nguồn không hợp lệ [[SOURCE:invented.pdf:9]]."
        )))
        chain = create_conversation_chain(RunnableLambda(lambda query: docs), llm)
        answer = chain.invoke("Câu hỏi")
        self.assertNotIn("[[SOURCE:", answer)
        self.assertNotIn("unused.pdf", answer)
        self.assertNotIn("invented.pdf", answer)
        self.assertEqual(answer.count("paper.pdf:2"), 1)

    def test_no_sources_for_uncited_answer(self):
        answer = "Tôi không biết dựa trên các tài liệu được cung cấp."
        self.assertEqual(format_answer_citations(answer), answer)


class StreamlitSessionTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.chain = Mock()
        self.chain.invoke.side_effect = lambda inputs: f"Trả lời: {inputs['question']}"
        self.builder = patch("chat.build_rag_chain", return_value=self.chain)
        self.build = self.builder.start()
        self.environment = patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
        self.environment.start()
        self.app = AppTest.from_file(str(BASE_DIR / "streamlit_app.py")).run(timeout=30)

    def tearDown(self):
        self.builder.stop()
        self.environment.stop()
        st.cache_resource.clear()

    def send(self, question):
        next(item for item in self.app.text_input if item.label == "Câu hỏi").set_value(question)
        next(item for item in self.app.button if item.label == "Gửi").click()
        self.app.run(timeout=30)
        self.assertFalse(self.app.exception)

    def test_multiple_turns_clear_history_and_reuse_pipeline(self):
        self.assertFalse(self.app.exception)
        for index in range(3):
            self.send(f"Câu hỏi {index}")
            self.assertEqual(len(self.app.session_state["messages"]), 2 * (index + 1))
            self.assertEqual(len(self.chain.invoke.call_args.args[0]["history"]), 2 * index)
        self.app.run(timeout=30)
        self.assertEqual(len(self.app.chat_message), 6)
        self.build.assert_called_once()
        next(item for item in self.app.button if item.label == "Xóa lịch sử chat").click().run()
        self.assertEqual(self.app.session_state["messages"], [])
        self.send("Bắt đầu lại")
        self.assertEqual(self.chain.invoke.call_args.args[0]["history"], [])
        self.build.assert_called_once()

    def test_request_failure_allows_next_message(self):
        self.chain.invoke.side_effect = [RuntimeError("thử lỗi"), "Đã trả lời"]
        self.send("Câu bị lỗi")
        self.assertTrue(self.app.error)
        self.assertEqual(self.app.session_state["messages"], [])
        self.send("Câu tiếp theo")
        self.assertFalse(self.app.error)
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertEqual(self.chain.invoke.call_args.args[0]["history"], [])

    def test_invalid_model_disables_send_and_keeps_history(self):
        self.send("Câu đầu")
        next(item for item in self.app.text_input if item.label == "Model Gemini").set_value(" ").run()
        self.assertTrue(next(item for item in self.app.button if item.label == "Gửi").disabled)
        self.assertEqual(len(self.app.session_state["messages"]), 2)

    def test_old_messages_display_short_sources_only_at_end(self):
        self.app.session_state["messages"] = [{
            "role": "assistant",
            "content": r"Ý một {C:\papers\paper.pdf:2}. Ý hai {C:\papers\paper.pdf:2}.",
        }]
        self.app.run(timeout=30)
        self.assertFalse(self.app.exception)
        rendered = self.app.chat_message[0].markdown[-1].value
        self.assertNotIn("C:\\", rendered)
        self.assertEqual(rendered.count("paper.pdf:2"), 1)
        self.assertIn("**Nguồn đã sử dụng**", rendered)


if __name__ == "__main__":
    unittest.main()
