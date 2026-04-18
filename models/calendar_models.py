from typing import List, Optional, Literal
from pydantic import BaseModel


# =========================
# ENUMS
# =========================
CalendarGroup = Literal[
    "travel", "work", "school", "kids", "social",
    "health", "fitness", "beauty", "finance", "family", "miscellaneous"
]

ReminderPriority = Literal["critical", "important", "light"]

ToneProfile = Literal["casual_warm", "efficient", "gentle", "upbeat"]

VisibilityMode = Literal[
    "private",
    "shared_household",
    "guardian_only",
    "selected_members_only",
    "summary_only"
]

HouseholdTaskType = Literal[
    "pickup", "drop", "payment", "gift", "documents",
    "school_event_attendance", "travel_prep",
    "medicine_pickup", "costume_prep", "airport_cab"
]

HouseholdTaskStatus = Literal[
    "unassigned", "assigned", "in_progress", "done"
]


# =========================
# CORE INPUT
# =========================
class CalendarEventInput(BaseModel):
    eventId: str
    title: str
    startAtISO: str
    endAtISO: Optional[str] = None
    timezone: Optional[str] = None
    venueName: Optional[str] = None
    venueAddress: Optional[str] = None
    notes: Optional[str] = None
    dressCode: Optional[str] = None
    linkedChildId: Optional[str] = None
    linkedChildName: Optional[str] = None
    ownerMemberId: Optional[str] = None
    autoPayEnabled: Optional[bool] = False
    amount: Optional[float] = None
    dueDateISO: Optional[str] = None
    participants: Optional[List[str]] = []


# =========================
# CLASSIFIED EVENT
# =========================
class ClassifiedEvent(CalendarEventInput):
    group: CalendarGroup
    subtype: str
    confidenceScore: float
    matchedSignals: List[str]
    missingFields: List[str]
    needsUserConfirmation: bool
    priority: ReminderPriority


# =========================
# REMINDER
# =========================
class Reminder(BaseModel):
    id: str
    offsetMinutes: int
    message: str
    priority: ReminderPriority
    toneProfile: ToneProfile
    sendAtISO: Optional[str] = None


# =========================
# CHECKLIST
# =========================
class ChecklistSection(BaseModel):
    title: str
    items: List[str]


class ChecklistBundle(BaseModel):
    carry: Optional[ChecklistSection] = None
    wear: Optional[ChecklistSection] = None
    prepTonight: Optional[ChecklistSection] = None
    documents: Optional[ChecklistSection] = None
    payment: Optional[ChecklistSection] = None
    childItems: Optional[ChecklistSection] = None


# =========================
# OUTFIT
# =========================
class OutfitPrompt(BaseModel):
    styleMode: str
    outfitKeywords: List[str]
    footwearKeywords: List[str]
    accessoryKeywords: List[str]
    notes: Optional[str] = None


# =========================
# PREDICTIVE OUTPUT
# =========================
class BufferPlan(BaseModel):
    prepNightBefore: bool
    startGettingReadyAtISO: Optional[str] = None
    leaveByISO: Optional[str] = None
    bufferReason: Optional[str] = None


class PredictiveOutput(BaseModel):
    prepTasks: List[str]
    packingList: List[str]
    linkedErrands: List[str]
    stressLoadScore: int
    outfitPrompt: Optional[OutfitPrompt] = None
    bufferPlan: Optional[BufferPlan] = None
    followupCandidates: List[str]


# =========================
# BRIEFING
# =========================
class BriefingSection(BaseModel):
    label: str
    lines: List[str]


class DayBriefing(BaseModel):
    type: Literal["morning_brief", "evening_prep_brief", "busy_day_rescue"]
    sections: List[BriefingSection]


# =========================
# FAMILY
# =========================
class FamilyMemberLite(BaseModel):
    memberId: str
    name: str
    role: Literal["primary_admin", "adult_member", "child_profile", "caregiver"]


class ResponsibilityItem(BaseModel):
    taskType: HouseholdTaskType
    memberId: Optional[str] = None
    status: HouseholdTaskStatus


class SharedEventOverlayLite(BaseModel):
    eventId: str
    ownerMemberId: str
    sharedWithMemberIds: List[str]
    visibilityMode: VisibilityMode
    responsibilityMap: List[ResponsibilityItem]


# =========================
# USER PREFERENCES
# =========================
class UserCalendarPreferences(BaseModel):
    toneProfile: Optional[ToneProfile] = None
    preferredReminderDensity: Optional[Literal["light", "balanced", "high_support"]] = "balanced"
    prefersShortMessages: Optional[bool] = False
    oftenRunsLate: Optional[bool] = False
    likesOutfitHelp: Optional[bool] = True
    packsNightBefore: Optional[bool] = False


# =========================
# FINAL RUNTIME RESULT
# =========================
class CalendarRuntimeResult(BaseModel):
    classifiedEvent: ClassifiedEvent
    predictiveOutput: PredictiveOutput
    checklistBundle: ChecklistBundle
    reminders: List[Reminder]
    dayBriefingHint: Optional[List[str]] = None