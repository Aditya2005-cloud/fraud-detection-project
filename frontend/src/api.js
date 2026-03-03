const BASE = import.meta.env.VITE_API_BASE || "";

export const api = {
  listTasks: async () => {
    const res = await fetch(BASE + "/api/tasks");
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  createTask: async (task) => {
    const res = await fetch(BASE + "/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(task),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  updateTask: async (id, task) => {
    const res = await fetch(BASE + `/api/tasks/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(task),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  deleteTask: async (id) => {
    const res = await fetch(BASE + `/api/tasks/${id}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error(await res.text());
    return null;
  },
};
