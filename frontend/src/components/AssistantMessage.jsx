// Assistant bubble, rendered exactly as CopilotKit's default except that the
// newest answer in the thread carries a speaker button in its toolbar.
//
// Only the latest one gets it: a thread should offer a single "play this"
// control, not one per message. CopilotKit already hides the whole toolbar
// while a message is still streaming, so the button can never appear on a
// half-written answer.
import { CopilotChatAssistantMessage } from "@copilotkit/react-core/v2";
import SpeakButton from "./SpeakButton.jsx";

export default function AssistantMessage(props) {
  const { message, messages } = props;

  // Last assistant message *with prose* — the tail of the list is often a
  // widget tool-call pair (service panel, comparison), which has no content
  // to speak.
  const lastSpokenId = [...(messages || [])]
    .reverse()
    .find((m) => m.role === "assistant" && m.content)?.id;

  const isLatest = Boolean(message?.id) && message.id === lastSpokenId;

  return (
    <CopilotChatAssistantMessage
      {...props}
      additionalToolbarItems={
        isLatest ? <SpeakButton text={String(message.content || "")} /> : undefined
      }
    />
  );
}
