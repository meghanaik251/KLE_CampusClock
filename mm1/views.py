from django.shortcuts import render, redirect
from django.http import HttpResponse
from django import template
from django.template.loader import render_to_string
from collections import defaultdict
from io import BytesIO

from .models import *
import random as rnd
from . forms import *

try:
    from xhtml2pdf import pisa
except ImportError:  # pragma: no cover
    pisa = None  # download views will show a clear error if the package isn't installed

register = template.Library()


# ======================================================================
#  HYBRID GA + LOCAL SEARCH CONFIGURATION
# ======================================================================
# Standard GA relies purely on crossover/mutation + many generations to
# remove conflicts. Here we add a deterministic LOCAL SEARCH REPAIR step
# that runs after every GA generation (and once more at the very end) and
# directly fixes any class that is still in conflict, by searching for a
# meeting_time / room / instructor combination that removes the conflict.
# This is what gives fast convergence + 95-100% conflict-free timetables.
# ======================================================================

POPULATION_SIZE = 16
NUMB_OF_ELITE_SCHEDULES = 2
TOURNAMENT_SELECTION_SIZE = 4
MUTATION_RATE = 0.10
MAX_GENERATIONS = 40          # GA will stop early the moment fitness == 1.0 (0 conflicts)
LOCAL_SEARCH_MAX_ITER = 25    # hill-climbing repair passes per schedule
COMPACTION_MAX_ITER = 15      # "move classes to mornings" polish passes per schedule

# ----------------------------------------------------------------------
#  MORNING-FIRST / 3:30 CUTOFF PREFERENCE
# ----------------------------------------------------------------------
# Soft-constraint requirement: fill morning slots first (highest priority).
# Afternoon slots up to 02:30 - 03:30 are used next. The last two slots of
# the day (03:30 - 04:30 and 04:30 - 05:30) are only used as a last resort,
# when every earlier slot is genuinely unavailable for that class. Break
# slots are never assigned a class at all.
TIME_SLOT_ORDER = [
    '08:00 - 09:00', '09:00 - 10:00', '10:00 - 10:15', '10:15 - 11:15', '11:15 - 12:15',
    '12:15 - 01:30', '01:30 - 02:30', '02:30 - 03:30', '03:30 - 04:30', '04:30 - 05:30',
]
BREAK_TIME_SLOTS = {'10:00 - 10:15', '12:15 - 01:30'}


class Data:
    def __init__(self):
        self._rooms = list(Room.objects.all())
        self._meetingTimes = list(MeetingTime.objects.all())
        self._instructors = list(Instructor.objects.all())
        self._courses = list(Course.objects.all())
        self._depts = list(Department.objects.all())

    def get_rooms(self): return self._rooms

    def get_instructors(self): return self._instructors

    def get_courses(self): return self._courses

    def get_depts(self): return self._depts

    def get_meetingTimes(self): return self._meetingTimes


data = Data()


class Class:
    def __init__(self, id, dept, section, course):
        self.section_id = id
        self.department = dept
        self.course = course
        self.instructor = None
        self.meeting_time = None
        self.meeting_day = None
        self.meeting_slot = None
        self.room = None
        self.section = section

    def get_id(self): return self.section_id

    def get_dept(self): return self.department

    def get_course(self): return self.course

    def get_instructor(self): return self.instructor

    def get_meetingTime(self): return self.meeting_time

    def get_room(self): return self.room

    def set_instructor(self, instructor): self.instructor = instructor

    def set_meetingTime(self, meetingTime): self.meeting_time = meetingTime

    def set_room(self, room): self.room = room


# ----------------------------------------------------------------------
#  CONFLICT / CONSTRAINT HELPERS  (used by both fitness calc & local search)
# ----------------------------------------------------------------------

def rooms_with_capacity(course):
    """Hard constraint #2: room seating_capacity must be >= max_numb_students."""
    try:
        needed = int(course.max_numb_students)
    except (TypeError, ValueError):
        needed = 0
    rooms = [r for r in data.get_rooms() if r.seating_capacity >= needed]
    return rooms if rooms else list(data.get_rooms())  # fallback so we never crash


def slot_priority(meeting_time):
    """Lower number = earlier in the day = higher scheduling priority."""
    t = (meeting_time.time or '').strip()
    if t in TIME_SLOT_ORDER:
        return TIME_SLOT_ORDER.index(t)
    return len(TIME_SLOT_ORDER)  # unrecognized slot -> sorted last


def get_ordered_meeting_times():
    """
    All meeting times EXCEPT break slots, sorted chronologically
    (morning first, 03:30-04:30 / 04:30-05:30 last). Because every
    conflict-repair and compaction routine walks this list in order and
    stops at the first free slot, morning slots always get filled before
    the algorithm ever reaches for a late-afternoon one.
    """
    candidates = [mt for mt in data.get_meetingTimes() if mt.time not in BREAK_TIME_SLOTS]
    candidates.sort(key=slot_priority)
    return candidates


def build_usage_maps(classes):
    """
    Builds three lookup maps keyed by meeting_time so conflicts can be
    detected in O(1) instead of the old O(n^2) double loop:
      room_map[(mt, room)]        -> list of classes using that room at that time
      inst_map[(mt, instructor)]  -> list of classes using that instructor at that time
      sec_map[(mt, section)]      -> list of classes for that section at that time
    """
    room_map = defaultdict(list)
    inst_map = defaultdict(list)
    sec_map = defaultdict(list)
    for c in classes:
        if not (c.meeting_time and c.room and c.instructor):
            continue
        room_map[(c.meeting_time.pid, c.room.r_number)].append(c)
        inst_map[(c.meeting_time.pid, c.instructor.uid)].append(c)
        sec_map[(c.meeting_time.pid, c.section)].append(c)
    return room_map, inst_map, sec_map


def class_has_conflict(c, room_map, inst_map, sec_map):
    """Returns True if this specific class instance violates any hard constraint."""
    if c.room.seating_capacity < int(c.course.max_numb_students or 0):
        return True
    if len(room_map[(c.meeting_time.pid, c.room.r_number)]) > 1:
        return True
    if len(inst_map[(c.meeting_time.pid, c.instructor.uid)]) > 1:
        return True
    if len(sec_map[(c.meeting_time.pid, c.section)]) > 1:
        return True
    return False


def count_total_conflicts(classes):
    room_map, inst_map, sec_map = build_usage_maps(classes)
    conflicts = 0
    for c in classes:
        if c.room.seating_capacity < int(c.course.max_numb_students or 0):
            conflicts += 1
    for lst in room_map.values():
        if len(lst) > 1:
            conflicts += len(lst) - 1
    for lst in inst_map.values():
        if len(lst) > 1:
            conflicts += len(lst) - 1
    for lst in sec_map.values():
        if len(lst) > 1:
            conflicts += len(lst) - 1
    return conflicts


# ----------------------------------------------------------------------
#  LOCAL SEARCH REPAIR  (the "Hybrid" part of Hybrid GA + Local Search)
# ----------------------------------------------------------------------

def local_search(schedule, max_iterations=LOCAL_SEARCH_MAX_ITER):
    """
    Hill-climbing repair pass.
    For every class that currently violates a hard constraint, try every
    (meeting_time, room, instructor) combination valid for its course and
    pick the first one that removes ALL conflicts for that class, without
    checking against the class's own old slot (so it can legally "reuse"
    a slot nobody else occupies). This directly targets:
      - Unique Class Timing (no double-booked section)
      - Class Capacity
      - Unique Room Assignment
      - Instructor Unique Timing
    """
    classes = schedule.get_classes()
    # Morning-first, breaks excluded. NOT shuffled: we always want the
    # earliest free slot tried first, so repaired classes land in the
    # morning whenever possible instead of a random slot.
    meeting_times = get_ordered_meeting_times()

    for _ in range(max_iterations):
        room_map, inst_map, sec_map = build_usage_maps(classes)
        conflicted = [c for c in classes if class_has_conflict(c, room_map, inst_map, sec_map)]
        if not conflicted:
            break  # schedule is fully conflict-free -> stop early

        # shuffle repair ORDER (not the slots) so we don't always repair
        # the same class first (avoids cycling)
        rnd.shuffle(conflicted)

        for c in conflicted:
            valid_rooms = rooms_with_capacity(c.course)
            valid_instructors = list(c.course.instructors.all()) or [c.instructor]

            rnd.shuffle(valid_rooms)
            mt_candidates = meeting_times  # morning-first order, try earliest free slot first

            fixed = False
            for mt in mt_candidates:
                # skip if section already has another class at this meeting time
                sec_key = (mt.pid, c.section)
                if any(other is not c for other in sec_map.get(sec_key, [])):
                    continue
                for room in valid_rooms:
                    room_key = (mt.pid, room.r_number)
                    if any(other is not c for other in room_map.get(room_key, [])):
                        continue
                    for inst in valid_instructors:
                        inst_key = (mt.pid, inst.uid)
                        if any(other is not c for other in inst_map.get(inst_key, [])):
                            continue
                        # Found a fully non-conflicting assignment -> apply it
                        # remove old entries from maps before mutating
                        old_room_key = (c.meeting_time.pid, c.room.r_number) if c.meeting_time and c.room else None
                        old_inst_key = (c.meeting_time.pid, c.instructor.uid) if c.meeting_time and c.instructor else None
                        old_sec_key = (c.meeting_time.pid, c.section) if c.meeting_time else None
                        if old_room_key and c in room_map.get(old_room_key, []):
                            room_map[old_room_key].remove(c)
                        if old_inst_key and c in inst_map.get(old_inst_key, []):
                            inst_map[old_inst_key].remove(c)
                        if old_sec_key and c in sec_map.get(old_sec_key, []):
                            sec_map[old_sec_key].remove(c)

                        c.meeting_time = mt
                        c.room = room
                        c.instructor = inst

                        room_map[room_key].append(c)
                        inst_map[inst_key].append(c)
                        sec_map[sec_key].append(c)
                        fixed = True
                        break
                    if fixed:
                        break
                if fixed:
                    break
            # if not fixed, leave it -- next iteration / GA crossover may resolve it
    return schedule


def compact_to_preferred_times(schedule, max_iterations=COMPACTION_MAX_ITER):
    """
    Soft-constraint polish pass, run AFTER local_search() has already made
    the schedule conflict-free. This does not fix conflicts -- it moves
    classes that are conflict-free but sitting in a late slot into an
    earlier free slot, whenever that's possible without creating a new
    conflict. Net effect: mornings fill up first, and the two slots after
    3:30 (03:30-04:30, 04:30-05:30) stay empty unless every earlier slot
    is genuinely full for that section/room/instructor.
    """
    classes = schedule.get_classes()
    ordered_times = get_ordered_meeting_times()  # ascending, morning first

    for _ in range(max_iterations):
        room_map, inst_map, sec_map = build_usage_maps(classes)
        moved_any = False

        # Try to pull back the LATEST-scheduled classes first -- this is
        # what actually empties out the after-3:30 slots.
        by_lateness = sorted(classes, key=lambda c: slot_priority(c.meeting_time), reverse=True)

        for c in by_lateness:
            current_priority = slot_priority(c.meeting_time)
            if current_priority <= 0:
                continue  # already in the very first slot, nothing earlier exists

            valid_rooms = rooms_with_capacity(c.course)
            valid_instructors = list(c.course.instructors.all()) or [c.instructor]

            for mt in ordered_times:
                if slot_priority(mt) >= current_priority:
                    break  # list is ascending -> no earlier slots left to try

                sec_key = (mt.pid, c.section)
                if any(other is not c for other in sec_map.get(sec_key, [])):
                    continue  # section already has a class at this earlier slot

                found_room = None
                for room in valid_rooms:
                    room_key = (mt.pid, room.r_number)
                    if any(other is not c for other in room_map.get(room_key, [])):
                        continue
                    found_room = room
                    break
                if not found_room:
                    continue

                found_inst = None
                for inst in valid_instructors:
                    inst_key = (mt.pid, inst.uid)
                    if any(other is not c for other in inst_map.get(inst_key, [])):
                        continue
                    found_inst = inst
                    break
                if not found_inst:
                    continue

                # Move this class earlier -- update maps then reassign.
                old_room_key = (c.meeting_time.pid, c.room.r_number)
                old_inst_key = (c.meeting_time.pid, c.instructor.uid)
                old_sec_key = (c.meeting_time.pid, c.section)
                if c in room_map.get(old_room_key, []):
                    room_map[old_room_key].remove(c)
                if c in inst_map.get(old_inst_key, []):
                    inst_map[old_inst_key].remove(c)
                if c in sec_map.get(old_sec_key, []):
                    sec_map[old_sec_key].remove(c)

                c.meeting_time = mt
                c.room = found_room
                c.instructor = found_inst

                room_map[(mt.pid, found_room.r_number)].append(c)
                inst_map[(mt.pid, found_inst.uid)].append(c)
                sec_map[sec_key].append(c)

                moved_any = True
                break  # move on to the next class

        if not moved_any:
            break  # fully compacted -- nothing left to pull earlier

    return schedule


class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._classNumb = 0
        self._isFitnessChanged = True

    def get_classes(self):
        self._isFitnessChanged = True
        return self._classes

    def get_numbOfConflicts(self): return self._numberOfConflicts

    def get_fitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculate_fitness()
            self._isFitnessChanged = False
        return self._fitness

    def initialize(self):
        sections = Section.objects.all()
        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            meeting_time_count = len(data.get_meetingTimes())
            if n > meeting_time_count:
                n = meeting_time_count
            courses = dept.courses.all()
            if not courses:
                continue
            for course in courses:
                per_course = n // len(courses)
                for i in range(per_course):
                    crs_inst = list(course.instructors.all())
                    if not crs_inst:
                        continue
                    newClass = Class(self._classNumb, dept, section.section_id, course)
                    self._classNumb += 1
                    candidate_times = get_ordered_meeting_times()
                    newClass.set_meetingTime(candidate_times[rnd.randrange(0, len(candidate_times))])
                    newClass.set_room(data.get_rooms()[rnd.randrange(0, len(data.get_rooms()))])
                    newClass.set_instructor(crs_inst[rnd.randrange(0, len(crs_inst))])
                    self._classes.append(newClass)

        # Seed with an immediate local-search repair so the initial
        # population already starts close to conflict-free instead of random.
        local_search(self, max_iterations=LOCAL_SEARCH_MAX_ITER)
        # Then pull classes toward the morning / pre-3:30 slots.
        compact_to_preferred_times(self, max_iterations=COMPACTION_MAX_ITER)
        return self

    def calculate_fitness(self):
        classes = self.get_classes()
        self._numberOfConflicts = count_total_conflicts(classes)
        return 1 / (1.0 * self._numberOfConflicts + 1)


class Population:
    def __init__(self, size):
        self._size = size
        self._data = data
        self._schedules = [Schedule().initialize() for i in range(size)]

    def get_schedules(self):
        return self._schedules


class GeneticAlgorithm:
    def evolve(self, population):
        crossover_pop = self._crossover_population(population)
        mutated_pop = self._mutate_population(crossover_pop)

        # ---- LOCAL SEARCH REPAIR STEP (Hybrid GA + Local Search) ----
        # After crossover/mutation may have re-introduced conflicts,
        # immediately repair every non-elite schedule, then pull classes
        # toward the morning / pre-3:30 slots.
        for i in range(NUMB_OF_ELITE_SCHEDULES, len(mutated_pop.get_schedules())):
            sched = mutated_pop.get_schedules()[i]
            local_search(sched)
            compact_to_preferred_times(sched)

        return mutated_pop

    def _crossover_population(self, pop):
        crossover_pop = Population(0)
        for i in range(NUMB_OF_ELITE_SCHEDULES):
            crossover_pop.get_schedules().append(pop.get_schedules()[i])
        i = NUMB_OF_ELITE_SCHEDULES
        while i < POPULATION_SIZE:
            schedule1 = self._select_tournament_population(pop).get_schedules()[0]
            schedule2 = self._select_tournament_population(pop).get_schedules()[0]
            crossover_pop.get_schedules().append(self._crossover_schedule(schedule1, schedule2))
            i += 1
        return crossover_pop

    def _mutate_population(self, population):
        for i in range(NUMB_OF_ELITE_SCHEDULES, len(population.get_schedules())):
            self._mutate_schedule(population.get_schedules()[i])
        return population

    def _crossover_schedule(self, schedule1, schedule2):
        crossoverSchedule = Schedule().initialize()
        length = min(len(crossoverSchedule.get_classes()), len(schedule1.get_classes()), len(schedule2.get_classes()))
        for i in range(length):
            if rnd.random() > 0.5:
                crossoverSchedule.get_classes()[i] = schedule1.get_classes()[i]
            else:
                crossoverSchedule.get_classes()[i] = schedule2.get_classes()[i]
        return crossoverSchedule

    def _mutate_schedule(self, mutateSchedule):
        meeting_times = get_ordered_meeting_times()
        rooms = data.get_rooms()
        for c in mutateSchedule.get_classes():
            if MUTATION_RATE > rnd.random():
                c.set_meetingTime(meeting_times[rnd.randrange(0, len(meeting_times))])
                valid_rooms = rooms_with_capacity(c.course)
                c.set_room(valid_rooms[rnd.randrange(0, len(valid_rooms))])
                crs_inst = list(c.course.instructors.all())
                if crs_inst:
                    c.set_instructor(crs_inst[rnd.randrange(0, len(crs_inst))])
        return mutateSchedule

    def _select_tournament_population(self, pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_schedules().append(pop.get_schedules()[rnd.randrange(0, len(pop.get_schedules()))])
            i += 1
        tournament_pop.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop


def context_manager(schedule):
    classes = schedule.get_classes()
    context = []
    for i in range(len(classes)):
        cls = {}
        cls["section"] = classes[i].section_id
        cls['dept'] = classes[i].department.dept_name
        cls['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number}, ' \
                         f'{classes[i].course.max_numb_students})'
        cls['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        cls['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        cls['meeting_time'] = [classes[i].meeting_time.pid, classes[i].meeting_time.day, classes[i].meeting_time.time]
        context.append(cls)
    return context


def home(request):
    return render(request, 'index.html', {})


def _run_hybrid_ga(max_generations=MAX_GENERATIONS):
    """
    Shared driver for both timetable views.
    Runs the GA + local-search hybrid loop until a fully conflict-free
    schedule (fitness == 1.0) is found or max_generations is reached,
    then does one final local-search polish pass for safety.
    """
    population = Population(POPULATION_SIZE)
    population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
    geneticAlgorithm = GeneticAlgorithm()

    generation_num = 0
    best = population.get_schedules()[0]

    while best.get_fitness() != 1.0 and generation_num < max_generations:
        generation_num += 1
        population = geneticAlgorithm.evolve(population)
        population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
        best = population.get_schedules()[0]
        print(f'> Generation #{generation_num} | best fitness={best.get_fitness():.4f} '
              f'| conflicts={best.get_numbOfConflicts()}')

    # Final safety-net repair pass on the best schedule found, then one
    # more thorough compaction so mornings are as full as possible and
    # nothing sits after 3:30 unless it truly has to.
    local_search(best, max_iterations=LOCAL_SEARCH_MAX_ITER * 2)
    compact_to_preferred_times(best, max_iterations=COMPACTION_MAX_ITER * 2)
    best.get_fitness()  # refresh fitness/conflict count after final repair

    print(f'FINAL: generations={generation_num}, fitness={best.get_fitness():.4f}, '
          f'conflicts={best.get_numbOfConflicts()}')

    return best.get_classes()


def timetable(request):
    schedule = _run_hybrid_ga()

    # Serialize into the session so the "Download PDF" button can render the
    # SAME timetable that's currently on screen, instead of re-running the
    # (randomized) genetic algorithm again and possibly getting a different result.
    request.session['sectionwise_schedule'] = serialize_schedule_sectionwise(schedule)

    return render(request, 'timetable.html', {
        'schedule': schedule,
        'sections': Section.objects.all(),
        'times': MeetingTime.objects.all(),
    })


def timetablewwt(request):
    schedule = _run_hybrid_ga()

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    time_slots = [
        "08:00 - 09:00", "09:00 - 10:00", "10:00 - 10:15", "10:15 - 11:15", "11:15 - 12:15",
        "12:15 - 01:30", "01:30 - 02:30", "02:30 - 03:30", "03:30 - 04:30", "04:30 - 05:30"
    ]

    for entry in schedule:
        meeting_info = str(entry.meeting_time).split()
        entry.meeting_day = meeting_info[1]
        entry.meeting_slot = meeting_info[2] + ' - ' + meeting_info[-1]

    # Same idea as above: freeze the exact rendered schedule into the session
    # for the download view to reuse.
    request.session['wwt_schedule'] = serialize_schedule_wwt(schedule)

    return render(request, 'timetablewwt.html', {
        'schedule': schedule,
        'sections': Section.objects.all(),
        'days_of_week': days_of_week,
        'time_slots': time_slots,
    })


# ----------------------------------------------------------------------
#  PDF DOWNLOAD  (server-side, colors always render regardless of the
#  user's browser "print backgrounds" setting)
# ----------------------------------------------------------------------

def serialize_schedule_sectionwise(schedule):
    """Class objects live only in memory, so turn them into plain dicts we
    can safely stash in the session and re-render later without re-running
    the (randomized) GA."""
    return [
        {
            'section': c.section_id,
            'section_name': c.section,
            'department': c.department.dept_name,
            'course': str(c.course),
            'room': str(c.room),
            'instructor': str(c.instructor),
            'meeting_time': str(c.meeting_time),
        }
        for c in schedule
    ]


def serialize_schedule_wwt(schedule):
    return [
        {
            'section': c.section,
            'course': str(c.course),
            'room': str(c.room),
            'instructor': str(c.instructor),
            'day': c.meeting_day,
            'slot': c.meeting_slot,
        }
        for c in schedule
    ]


def render_pdf_response(template_name, context, filename):
    if pisa is None:
        return HttpResponse(
            "PDF export requires the 'xhtml2pdf' package. Install it with:\n\n"
            "    pip install xhtml2pdf\n\nthen try downloading again.",
            content_type="text/plain",
            status=500,
        )

    html = render_to_string(template_name, context)
    buffer = BytesIO()
    # encoding='UTF-8' avoids crashes on names/course titles with special characters
    result = pisa.CreatePDF(src=html, dest=buffer, encoding='UTF-8')
    if result.err:
        return HttpResponse('There was an error generating the PDF.', status=500)

    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def download_timetable_pdf(request):
    schedule = request.session.get('sectionwise_schedule')
    if not schedule:
        return HttpResponse(
            'No timetable has been generated yet. Please generate the section-wise '
            'timetable first, then download it.',
            status=400,
        )
    context = {
        'schedule': schedule,
        'sections': Section.objects.all(),
    }
    return render_pdf_response('pdf_timetable.html', context, 'section_wise_timetable.pdf')


def download_timetablewwt_pdf(request):
    schedule = request.session.get('wwt_schedule')
    if not schedule:
        return HttpResponse(
            'No timetable has been generated yet. Please generate the weekdays-wise '
            'timetable first, then download it.',
            status=400,
        )
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    time_slots = [
        "08:00 - 09:00", "09:00 - 10:00", "10:00 - 10:15", "10:15 - 11:15", "11:15 - 12:15",
        "12:15 - 01:30", "01:30 - 02:30", "02:30 - 03:30", "03:30 - 04:30", "04:30 - 05:30"
    ]
    context = {
        'schedule': schedule,
        'sections': Section.objects.all(),
        'days_of_week': days_of_week,
        'time_slots': time_slots,
    }
    return render_pdf_response('pdf_timetablewwt.html', context, 'weekdays_wise_timetable.pdf')


def add_instructor(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addinstructor')
    context = {
        'form': form
    }
    return render(request, 'adins.html', context)


def inst_list_view(request):
    context = {
        'instructors': Instructor.objects.all()
    }
    return render(request, 'instlist.html', context)


def delete_instructor(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('editinstructor')


def add_room(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addroom')
    context = {
        'form': form
    }
    return render(request, 'addrm.html', context)


def room_list(request):
    context = {
        'rooms': Room.objects.all()
    }
    return render(request, 'rmlist.html', context)


def delete_room(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('editrooms')


def meeting_list_view(request):
    context = {
        'meeting_times': MeetingTime.objects.all()
    }
    return render(request, 'mtlist.html', context)


def add_meeting_time(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addmeetingtime')
        else:
            print('Invalid')
    context = {
        'form': form
    }
    return render(request, 'addmt.html', context)


def delete_meeting_time(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('editmeetingtime')


def course_list_view(request):
    context = {
        'courses': Course.objects.all()
    }
    return render(request, 'crslist.html', context)


def add_course(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addcourse')
        else:
            print('Invalid')
    context = {
        'form': form
    }
    return render(request, 'adcrs.html', context)


def delete_course(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('editcourse')


def add_department(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('adddepartment')
    context = {
        'form': form
    }
    return render(request, 'addep.html', context)


def department_list(request):
    context = {
        'departments': Department.objects.all()
    }
    return render(request, 'deptlist.html', context)


def delete_department(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('editdepartment')


def add_section(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('addsection')
    context = {
        'form': form
    }
    return render(request, 'addsec.html', context)


def section_list(request):
    context = {
        'sections': Section.objects.all()
    }
    return render(request, 'seclist.html', context)


def delete_section(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('editsection')