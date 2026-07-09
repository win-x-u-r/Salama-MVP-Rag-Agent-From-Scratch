export default function Message({ message }) {
  const { role, content, pending } = message;
  return (
    <div className={`msg ${role}`}>
      <div className="avatar">{role === "user" ? "You" : "AI"}</div>
      <div className="bubble">
        {pending ? (
          <span className="typing">
            <span />
            <span />
            <span />
          </span>
        ) : (
          content
        )}
      </div>
    </div>
  );
}
