import React from "react";
import TaskItem from "./TaskItems";

const columns = [
  {
    key: "todo",
    title: "Alerts",
    accentClass: "board-column--alerts",
  },
  {
    key: "doing",
    title: "Investigating",
    accentClass: "board-column--investigating",
  },
  {
    key: "done",
    title: "Cleared",
    accentClass: "board-column--cleared",
  },
];

export default function TaskList({ tasks = [], onEdit, onDelete }) {
  const grouped = columns.map((col) => ({
    ...col,
    items: tasks.filter((t) => (t.status || "todo") === col.key),
  }));

  return (
    <div className="board-grid">
      {grouped.map((col) => (
        <div
          key={col.key}
          className={`board-column ${col.accentClass}`}
        >
          <div className="column-header">
            <span className="column-title">{col.title}</span>
            <span className="badge column-count">{col.items.length}</span>
          </div>

          {!col.items.length ? (
            <div className="column-empty">No cards yet. Drop one here!</div>
          ) : (
            <div className="column-cards">
              {col.items.map((t) => (
                <TaskItem
                  key={t.id || t._id}
                  task={t}
                  onEdit={onEdit}
                  onDelete={onDelete}
                />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
