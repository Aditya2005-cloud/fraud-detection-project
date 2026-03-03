import React, { useEffect, useState } from "react";

const statusOptions = [
  { value: "todo", label: "To do" },
  { value: "doing", label: "Doing" },
  { value: "done", label: "Done" },
];

export default function TaskForm({ onSubmit, initial = null, onCancel }) {
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [status, setStatus] = useState("todo");
  const [tagsText, setTagsText] = useState("");

  useEffect(() => {
    if (initial) {
      setTitle(initial.title || "");
      setDesc(initial.description || "");
      setStatus(initial.status || "todo");
      setTagsText(Array.isArray(initial.tags) ? initial.tags.join(", ") : "");
    } else {
      setTitle("");
      setDesc("");
      setStatus("todo");
      setTagsText("");
    }
  }, [initial]);

  function submit(e) {
    e.preventDefault();
    if (!title.trim()) return;
    const tags =
      tagsText
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean) || [];
    onSubmit({
      title: title.trim(),
      description: desc.trim(),
      status,
      tags,
    });
  }

  return (
    <form onSubmit={submit} className="task-form">
      <input
        className="input-field"
        placeholder="Title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        className="input-field textarea-field"
        placeholder="Description (optional)"
        value={desc}
        onChange={(e) => setDesc(e.target.value)}
      />
      <div className="form-grid">
        <div className="form-field">
          <label className="field-label">Status</label>
          <select
            className="input-field select-field"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
          >
            {statusOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div className="form-field">
          <label className="field-label">Tags (comma separated)</label>
          <input
            className="input-field"
            placeholder="e.g. HTML, CSS, React"
            value={tagsText}
            onChange={(e) => setTagsText(e.target.value)}
          />
        </div>
      </div>
      <div className="form-actions">
        <button className="btn-primary animate-bounce-soft">Save</button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="btn-secondary"
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
