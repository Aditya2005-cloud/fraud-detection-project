import React from "react";

const statusStyles = {
  todo: "status-todo",
  doing: "status-doing",
  done: "status-done",
};

export default function TaskItem({ task, onEdit, onDelete }) {
  const tags = Array.isArray(task.tags) ? task.tags : [];
  const status = task.status || "todo";

  return (
    <div className="task-card">
      <div className="task-top">
        <span className={`badge ${statusStyles[status] || "status-fallback"}`}>
          {status === "todo" ? "To do" : status === "doing" ? "Doing" : "Done"}
        </span>
        <div className="task-actions">
          <button onClick={() => onEdit(task)} className="btn-secondary btn-compact">
            Edit
          </button>
          <button
            onClick={() => onDelete(task)}
            className="btn-primary btn-danger"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="task-body">
        <h3 className="task-title">{task.title}</h3>
        {task.description && <p className="task-desc">{task.description}</p>}
      </div>

      {!!tags.length && (
        <div className="task-tags">
          {tags.map((tag) => (
            <span key={tag} className="tag-pill">
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
