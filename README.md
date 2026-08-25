# KLE CampusClock

**Intelligent Timetable Generator powered by Genetic Algorithms & Local Search**

Built on Django + SQLite, CampusClock automatically assigns courses, instructors, rooms, and time slots across sections while satisfying scheduling constraints — no manual timetable juggling required.

---

## ✨ Features

- **Hybrid GA + Local Search scheduling engine** — genetic algorithm evolves candidate timetables, while a deterministic local-search repair pass fixes conflicts immediately instead of waiting on random mutation
- **Two timetable views**: Section-wise and Weekdays-wise, each downloadable
- **Server-side PDF export** with full-color output, matching what's on screen
- **Morning-first slot allocation** — mornings fill up before afternoon slots; the two last slots of the day (after 3:30 PM) are only used when nothing earlier is available
- **Full CRUD admin forms** for Rooms, Instructors, Meeting Times, Courses, Departments, and Sections
- **Conflict-free guarantee** (subject to data feasibility — see [Constraints](#constraints-handled) below)

---

## 🧠 How Scheduling Works

Unlike a standard GA that relies purely on crossover/mutation across many generations, this project repairs conflicts directly:

1. **Initialize** — build a starting population of random (but course/department-valid) timetables
2. **Evaluate fitness** — `fitness = 1 / (1 + total_conflicts)`; a perfectly conflict-free schedule scores `1.0`
3. **Evolve** — tournament selection → crossover → mutation, same as classic GA
4. **Local Search Repair** *(the "hybrid" part)* — after every generation, any class still in conflict is reassigned to the first valid `(meeting_time, room, instructor)` combination that clears it
5. **Compaction** — a second pass nudges conflict-free classes toward earlier slots, so the schedule stays front-loaded in the morning rather than randomly scattered across the day
6. **Repeat** until fitness hits `1.0` or a generation cap is reached, then one final repair + compaction pass before rendering

---

## Constraints Handled

### Hard Constraints (must never be violated)

| # | Constraint | Enforced by |
|---|---|---|
| 1 | **Unique class timing** — a section never has two classes at the same time | `sec_map[(meeting_time, section)]` conflict check + local search reassignment |
| 2 | **Room capacity** — room seating capacity ≥ course's max students | Rooms below capacity are filtered out of candidates before assignment even happens |
| 3 | **Unique room assignment** — no two classes share a room at the same time | `room_map[(meeting_time, room)]` conflict check + repair |
| 4 | **Instructor's unique timing** — no instructor teaches two classes simultaneously | `inst_map[(meeting_time, instructor)]` conflict check + repair |
| 5 | **Instructor–course validity** — a class is only ever assigned an instructor registered to teach that course | Instructor candidates are drawn from `course.instructors.all()`, never the full instructor pool |
| 6 | **No classes during break slots** | Break slots are excluded entirely from the candidate slot list |

### Soft Constraints (optimized where possible, not strictly enforced)

| Constraint | Status |
|---|---|
| **Departmental alignment** — courses scheduled only within their own department | ✅ Implicit — sections only draw from `department.courses.all()` |
| **Morning-first / post-3:30 avoidance** — fill morning slots before afternoon; avoid the last two slots of the day unless unavoidable | ✅ Implemented via the compaction pass |
| **Balanced weekly distribution** — spread a section's classes evenly across the week rather than clustering them | ⏳ Not yet optimized — local search takes the first valid slot, not the most balanced one |
| **Section-specific time preferences** | ⏳ Not yet modeled — no preference field currently exists on `Section` |

> Note: 100% conflict-free output assumes your input data is *feasible* — e.g. enough rooms/instructors/slots exist to fit every section's weekly class count. If it isn't, the engine will get as close as mathematically possible and report the remaining conflict count.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.6+, Django 2.0+ |
| Database | SQLite3 |
| Scheduling Engine | Custom Hybrid Genetic Algorithm + Local Search |
| PDF Export | [`xhtml2pdf`](https://github.com/xhtml2pdf/xhtml2pdf) |
| Frontend | Django Templates, HTML/CSS |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.6 or above
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/meghanaik251/KLE_CampusClock.git
cd KLE-CampusClock/M1

# 2. pip install

# 3. Run the development server
python manage.py runserver
```

Then open your browser to:

| Purpose | URL |
|---|---|
| Home | `http://127.0.0.1:8000/` |
| Section-wise timetable | `http://127.0.0.1:8000/timetable_generation/` |
| Weekdays-wise timetable | `http://127.0.0.1:8000/timetable_generationwwt/` |
| Django admin | `http://127.0.0.1:8000/admin/` |

### Usage Flow
1. Add your **Rooms**, **Instructors**, **Meeting Times**, **Courses**, **Departments**, and **Sections** via the nav bar forms
2. Click **Generate Timetable** (section-wise or weekdays-wise)
3. Review the generated, conflict-free schedule
4. Click **Download PDF** to export a full-color copy


---

## 🗺️ Roadmap

- [ ] Balanced weekly distribution as a secondary local-search objective
- [ ] Per-section time preferences (e.g. "prefer mornings" flag)
- [ ] Faculty workload balancing across the week
- [ ] Real-time conflict resolution when editing an existing timetable
- [ ] REST API for external integrations

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue first for any significant change so it can be discussed before implementation.

---

## 📬 Contact

**Meghana G Naik** — [meghanaiktech@gmail.com](mailto:meghanaiktech@gmail.com)
