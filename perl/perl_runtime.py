#!/usr/bin/env python3

import os
import sys
import gi
import json
import csv
import time
import math
import argparse
import platform
import traceback
from datetime import datetime

gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib

import pyds


PERSON = 0
HELMET = 1
VEST = 2

UNTRACKED_ID = 0xFFFFFFFFFFFFFFFF


# ============================================================
# Geometry
# ============================================================

def bbox_area(b):
    return (
        max(0.0, b[2])
        *
        max(0.0, b[3])
    )


def bbox_center(b):

    x, y, w, h = b

    return (
        x + w * 0.5,
        y + h * 0.5
    )


def bbox_bottom_center(b):

    x, y, w, h = b

    return (
        x + w * 0.5,
        y + h
    )


def intersection_area(a, b):

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    return iw * ih


def point_in_polygon(
    point,
    polygon
):

    x, y = point

    inside = False

    j = len(polygon) - 1

    for i in range(
        len(polygon)
    ):

        xi, yi = polygon[i]
        xj, yj = polygon[j]

        crosses = (
            (yi > y)
            !=
            (yj > y)
        )

        if crosses:

            denom = yj - yi

            if abs(denom) < 1e-9:
                denom = 1e-9

            x_cross = (
                (xj - xi)
                *
                (y - yi)
                /
                denom
                +
                xi
            )

            if x < x_cross:
                inside = not inside

        j = i

    return inside


# ============================================================
# PPE association
# ============================================================

def match_ppe(
    persons,
    detections,
    kind,
    cfg
):

    if kind == "helmet":

        min_containment = cfg[
            "helmet_min_containment"
        ]

    else:

        min_containment = cfg[
            "vest_min_containment"
        ]


    candidates = []

    for person_index, person in enumerate(
        persons
    ):

        px, py, pw, ph = (
            person["bbox"]
        )

        if pw <= 1 or ph <= 1:
            continue

        for det_index, det in enumerate(
            detections
        ):

            area = bbox_area(
                det["bbox"]
            )

            if area <= 1:
                continue

            containment = (
                intersection_area(
                    person["bbox"],
                    det["bbox"]
                )
                /
                area
            )

            if (
                containment
                <
                min_containment
            ):
                continue

            cx, cy = bbox_center(
                det["bbox"]
            )

            rel_y = (
                cy - py
            ) / ph


            if kind == "helmet":

                if (
                    rel_y
                    >
                    cfg[
                        "helmet_region_max"
                    ]
                ):
                    continue

            else:

                if (
                    rel_y
                    <
                    cfg[
                        "vest_region_min"
                    ]
                ):
                    continue

                if (
                    rel_y
                    >
                    cfg[
                        "vest_region_max"
                    ]
                ):
                    continue


            person_cx = (
                px
                +
                pw * 0.5
            )

            horizontal_error = abs(
                cx - person_cx
            ) / max(
                pw * 0.5,
                1.0
            )

            horizontal_score = max(
                0.0,
                1.0
                -
                horizontal_error
            )

            score = (
                0.80
                *
                containment
                +
                0.20
                *
                horizontal_score
            )

            candidates.append(
                (
                    score,
                    person_index,
                    det_index
                )
            )


    candidates.sort(
        reverse=True
    )

    used_persons = set()
    used_detections = set()

    result = {}

    for (
        score,
        person_index,
        det_index
    ) in candidates:

        if (
            person_index
            in
            used_persons
        ):
            continue

        if (
            det_index
            in
            used_detections
        ):
            continue

        track_id = persons[
            person_index
        ][
            "track_id"
        ]

        result[
            track_id
        ] = {
            "score":
                score,

            "det":
                detections[
                    det_index
                ]
        }

        used_persons.add(
            person_index
        )

        used_detections.add(
            det_index
        )

    return result


# ============================================================
# Event writer
# ============================================================

class EventWriter:

    def __init__(
        self,
        event_dir
    ):

        self.event_dir = (
            event_dir
        )

        os.makedirs(
            event_dir,
            exist_ok=True
        )

        self.jsonl_path = (
            os.path.join(
                event_dir,
                "events.jsonl"
            )
        )

        self.csv_path = (
            os.path.join(
                event_dir,
                "events.csv"
            )
        )

        self.active = {}

        self.counter = 0

        self.csv_fp = open(
            self.csv_path,
            "a",
            newline=""
        )

        self.csv_writer = (
            csv.writer(
                self.csv_fp
            )
        )

        if (
            os.path.getsize(
                self.csv_path
            )
            ==
            0
        ):

            self.csv_writer.writerow([
                "system_time",
                "video_time_sec",
                "event_id",
                "action",
                "event_type",
                "track_id",
                "zone",
                "details_json"
            ])

            self.csv_fp.flush()


    def _event_id(self):

        self.counter += 1

        return "{}_{:05d}".format(
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )[:-3],
            self.counter
        )


    def _write(
        self,
        action,
        event,
        now,
        details=None
    ):

        if details is None:
            details = (
                event.get(
                    "details",
                    {}
                )
            )

        record = {
            "system_time":
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),

            "video_time_sec":
                round(
                    now,
                    3
                ),

            "event_id":
                event[
                    "event_id"
                ],

            "action":
                action,

            "event_type":
                event[
                    "event_type"
                ],

            "track_id":
                event.get(
                    "track_id"
                ),

            "zone":
                event.get(
                    "zone"
                ),

            "details":
                details
        }

        with open(
            self.jsonl_path,
            "a"
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                +
                "\n"
            )

        self.csv_writer.writerow([
            record[
                "system_time"
            ],

            record[
                "video_time_sec"
            ],

            record[
                "event_id"
            ],

            record[
                "action"
            ],

            record[
                "event_type"
            ],

            record[
                "track_id"
            ],

            record[
                "zone"
            ],

            json.dumps(
                record[
                    "details"
                ],
                ensure_ascii=False
            )
        ])

        self.csv_fp.flush()

        print(
            "[EVENT]"
            " action={}"
            " type={}"
            " track={}"
            " zone={}".format(
                action,
                record[
                    "event_type"
                ],
                record[
                    "track_id"
                ],
                record[
                    "zone"
                ]
            )
        )


    def trigger(
        self,
        event_type,
        track_id,
        zone,
        now,
        details=None
    ):

        event = {
            "event_id":
                self._event_id(),

            "event_type":
                event_type,

            "track_id":
                track_id,

            "zone":
                zone,

            "details":
                details or {}
        }

        self._write(
            "TRIGGER",
            event,
            now,
            details
        )


    def open_event(
        self,
        key,
        event_type,
        track_id,
        zone,
        now,
        details=None
    ):

        current = self.active.get(
            key
        )

        if current is not None:

            if (
                current[
                    "event_type"
                ]
                ==
                event_type
            ):
                return

            self.close_event(
                key,
                now,
                {
                    "reason":
                        "state_changed"
                }
            )

        event = {
            "event_id":
                self._event_id(),

            "event_type":
                event_type,

            "track_id":
                track_id,

            "zone":
                zone,

            "start_time":
                now,

            "details":
                details or {}
        }

        self.active[
            key
        ] = event

        self._write(
            "OPEN",
            event,
            now,
            details
        )


    def close_event(
        self,
        key,
        now,
        details=None
    ):

        event = self.active.pop(
            key,
            None
        )

        if event is None:
            return

        merged = {
            "duration_sec":
                round(
                    now
                    -
                    event[
                        "start_time"
                    ],
                    3
                )
        }

        if details:
            merged.update(
                details
            )

        self._write(
            "CLOSE",
            event,
            now,
            merged
        )


    def close_track(
        self,
        track_id,
        now
    ):

        keys = []

        for key, event in (
            self.active.items()
        ):

            if (
                event.get(
                    "track_id"
                )
                ==
                track_id
            ):

                keys.append(
                    key
                )

        for key in keys:

            self.close_event(
                key,
                now,
                {
                    "reason":
                        "track_lost"
                }
            )


    def shutdown(
        self,
        now
    ):

        for key in list(
            self.active.keys()
        ):

            self.close_event(
                key,
                now,
                {
                    "reason":
                        "pipeline_end"
                }
            )

        self.csv_fp.close()


# ============================================================
# PERL engine
# ============================================================

class PERLEngine:

    def __init__(
        self,
        case_cfg,
        events
    ):

        self.cfg = case_cfg

        self.case_id = (
            case_cfg[
                "case_id"
            ]
        )

        self.zone_name = (
            case_cfg[
                "display_name"
            ]
        )

        self.polygon = [
            tuple(p)
            for p in
            case_cfg[
                "polygon_pixel"
            ]
        ]

        self.rules = (
            case_cfg[
                "rules"
            ]
        )

        self.timing = (
            case_cfg[
                "timing"
            ]
        )

        self.ppe_cfg = (
            case_cfg[
                "ppe_matching"
            ]
        )

        self.events = events

        self.tracks = {}

        self.over_since = None
        self.clear_since = None

        self.last_time = 0.0


    def get_track(
        self,
        track_id,
        now
    ):

        if track_id not in self.tracks:

            self.tracks[
                track_id
            ] = {
                "created":
                    now,

                "last_seen":
                    now,

                "inside":
                    False,

                "raw_inside_since":
                    None,

                "raw_outside_since":
                    None,

                "enter_time":
                    None,

                "helmet":
                    None,

                "vest":
                    None,

                "last_helmet_seen":
                    None,

                "last_vest_seen":
                    None,

                "helmet_recover_since":
                    None,

                "vest_recover_since":
                    None
            }

        state = (
            self.tracks[
                track_id
            ]
        )

        state[
            "last_seen"
        ] = now

        return state


    def reset_ppe(
        self,
        state
    ):

        state[
            "helmet"
        ] = None

        state[
            "vest"
        ] = None

        state[
            "last_helmet_seen"
        ] = None

        state[
            "last_vest_seen"
        ] = None

        state[
            "helmet_recover_since"
        ] = None

        state[
            "vest_recover_since"
        ] = None


    def update_zone_state(
        self,
        track_id,
        state,
        raw_inside,
        now
    ):

        transition = None

        if raw_inside:

            state[
                "raw_outside_since"
            ] = None

            if (
                state[
                    "raw_inside_since"
                ]
                is None
            ):

                state[
                    "raw_inside_since"
                ] = now

            if (
                not
                state[
                    "inside"
                ]
                and
                now
                -
                state[
                    "raw_inside_since"
                ]
                >=
                self.timing[
                    "enter_confirm_sec"
                ]
            ):

                state[
                    "inside"
                ] = True

                state[
                    "enter_time"
                ] = now

                transition = "ENTER"

                self.events.trigger(
                    "ZONE_ENTRY",
                    track_id,
                    self.zone_name,
                    now
                )

        else:

            state[
                "raw_inside_since"
            ] = None

            if (
                state[
                    "raw_outside_since"
                ]
                is None
            ):

                state[
                    "raw_outside_since"
                ] = now

            if (
                state[
                    "inside"
                ]
                and
                now
                -
                state[
                    "raw_outside_since"
                ]
                >=
                self.timing[
                    "exit_confirm_sec"
                ]
            ):

                state[
                    "inside"
                ] = False

                state[
                    "enter_time"
                ] = None

                transition = "EXIT"

                self.events.close_event(
                    (
                        "ppe",
                        track_id
                    ),
                    now,
                    {
                        "reason":
                            "left_zone"
                    }
                )

                self.events.close_event(
                    (
                        "dwell",
                        track_id
                    ),
                    now,
                    {
                        "reason":
                            "left_zone"
                    }
                )

                self.events.trigger(
                    "ZONE_EXIT",
                    track_id,
                    self.zone_name,
                    now
                )

                self.reset_ppe(
                    state
                )

        return transition


    def update_ppe_state(
        self,
        state,
        kind,
        detected,
        now
    ):

        value_key = kind

        last_key = (
            "last_{}_seen"
        ).format(
            kind
        )

        recover_key = (
            "{}_recover_since"
        ).format(
            kind
        )

        missing_sec = (
            self.timing[
                "ppe_missing_confirm_sec"
            ]
        )

        recover_sec = (
            self.timing[
                "ppe_recover_confirm_sec"
            ]
        )


        if detected:

            state[
                last_key
            ] = now

            if (
                state[
                    value_key
                ]
                is False
            ):

                if (
                    state[
                        recover_key
                    ]
                    is None
                ):

                    state[
                        recover_key
                    ] = now

                elif (
                    now
                    -
                    state[
                        recover_key
                    ]
                    >=
                    recover_sec
                ):

                    state[
                        value_key
                    ] = True

            else:

                state[
                    value_key
                ] = True

                state[
                    recover_key
                ] = now


        else:

            state[
                recover_key
            ] = None

            base = state[
                last_key
            ]

            if base is None:

                base = state[
                    "enter_time"
                ]

            if base is None:

                base = state[
                    "created"
                ]

            if (
                now - base
                >=
                missing_sec
            ):

                state[
                    value_key
                ] = False


    def ppe_issues(
        self,
        state
    ):

        rule = self.rules[
            "ppe"
        ]

        if not rule[
            "enabled"
        ]:

            return []

        issues = []

        if (
            rule[
                "helmet_required"
            ]
            and
            state[
                "helmet"
            ]
            is False
        ):

            issues.append(
                "NO HELMET"
            )

        if (
            rule[
                "vest_required"
            ]
            and
            state[
                "vest"
            ]
            is False
        ):

            issues.append(
                "NO VEST"
            )

        return issues


    def update_occupancy(
        self,
        count,
        now
    ):

        rule = self.rules[
            "occupancy"
        ]

        if not rule[
            "enabled"
        ]:

            return False

        max_people = int(
            rule[
                "max_people"
            ]
        )

        key = (
            "occupancy",
            self.case_id
        )

        if count > max_people:

            self.clear_since = None

            if (
                self.over_since
                is None
            ):

                self.over_since = now

            if (
                now
                -
                self.over_since
                >=
                self.timing[
                    "overcrowd_confirm_sec"
                ]
            ):

                self.events.open_event(
                    key,
                    "OVERCROWDING",
                    None,
                    self.zone_name,
                    now,
                    {
                        "persons":
                            count,

                        "max_people":
                            max_people
                    }
                )

        else:

            self.over_since = None

            if (
                self.clear_since
                is None
            ):

                self.clear_since = now

            if (
                key
                in
                self.events.active
                and
                now
                -
                self.clear_since
                >=
                self.timing[
                    "overcrowd_clear_sec"
                ]
            ):

                self.events.close_event(
                    key,
                    now,
                    {
                        "persons":
                            count,

                        "max_people":
                            max_people
                    }
                )

        return (
            key
            in
            self.events.active
        )


    def process(
        self,
        persons,
        helmets,
        vests,
        now
    ):

        self.last_time = now

        inside_persons = []

        results = {}

        for person in persons:

            track_id = (
                person[
                    "track_id"
                ]
            )

            if (
                track_id
                ==
                UNTRACKED_ID
            ):

                continue

            state = self.get_track(
                track_id,
                now
            )

            raw_inside = (
                point_in_polygon(
                    bbox_bottom_center(
                        person[
                            "bbox"
                        ]
                    ),
                    self.polygon
                )
            )

            self.update_zone_state(
                track_id,
                state,
                raw_inside,
                now
            )

            if state[
                "inside"
            ]:

                inside_persons.append(
                    person
                )


        ppe_enabled = (
            self.rules[
                "ppe"
            ][
                "enabled"
            ]
        )

        if ppe_enabled:

            helmet_map = (
                match_ppe(
                    inside_persons,
                    helmets,
                    "helmet",
                    self.ppe_cfg
                )
            )

            vest_map = (
                match_ppe(
                    inside_persons,
                    vests,
                    "vest",
                    self.ppe_cfg
                )
            )

        else:

            helmet_map = {}
            vest_map = {}


        any_person_warning = False

        for person in inside_persons:

            track_id = (
                person[
                    "track_id"
                ]
            )

            state = (
                self.tracks[
                    track_id
                ]
            )


            if ppe_enabled:

                self.update_ppe_state(
                    state,
                    "helmet",
                    (
                        track_id
                        in
                        helmet_map
                    ),
                    now
                )

                self.update_ppe_state(
                    state,
                    "vest",
                    (
                        track_id
                        in
                        vest_map
                    ),
                    now
                )


            issues = (
                self.ppe_issues(
                    state
                )
            )


            if issues:

                any_person_warning = True

                if (
                    "NO HELMET"
                    in
                    issues
                    and
                    "NO VEST"
                    in
                    issues
                ):

                    event_type = (
                        "NO_HELMET_AND_VEST"
                    )

                elif (
                    "NO HELMET"
                    in
                    issues
                ):

                    event_type = (
                        "NO_HELMET"
                    )

                else:

                    event_type = (
                        "NO_VEST"
                    )

                self.events.open_event(
                    (
                        "ppe",
                        track_id
                    ),
                    event_type,
                    track_id,
                    self.zone_name,
                    now,
                    {
                        "issues":
                            issues
                    }
                )

            else:

                self.events.close_event(
                    (
                        "ppe",
                        track_id
                    ),
                    now,
                    {
                        "reason":
                            "ppe_compliant"
                    }
                )


            dwell_sec = 0.0

            overstay = False

            dwell_rule = (
                self.rules[
                    "dwell"
                ]
            )

            if (
                state[
                    "enter_time"
                ]
                is not None
            ):

                dwell_sec = (
                    now
                    -
                    state[
                        "enter_time"
                    ]
                )


            if (
                dwell_rule[
                    "enabled"
                ]
                and
                dwell_sec
                >=
                float(
                    dwell_rule[
                        "limit_sec"
                    ]
                )
            ):

                overstay = True

                any_person_warning = True

                self.events.open_event(
                    (
                        "dwell",
                        track_id
                    ),
                    "OVERSTAY",
                    track_id,
                    self.zone_name,
                    now,
                    {
                        "dwell_sec":
                            round(
                                dwell_sec,
                                3
                            ),

                        "limit_sec":
                            float(
                                dwell_rule[
                                    "limit_sec"
                                ]
                            )
                    }
                )


            observing = False

            if ppe_enabled:

                required_states = []

                if (
                    self.rules[
                        "ppe"
                    ][
                        "helmet_required"
                    ]
                ):
                    required_states.append(
                        state[
                            "helmet"
                        ]
                    )

                if (
                    self.rules[
                        "ppe"
                    ][
                        "vest_required"
                    ]
                ):
                    required_states.append(
                        state[
                            "vest"
                        ]
                    )

                observing = any(
                    v is None
                    for v
                    in
                    required_states
                )


            if (
                issues
                or
                overstay
            ):

                status = "WARNING"

            elif observing:

                status = "OBSERVING"

            else:

                status = "SAFE"


            results[
                track_id
            ] = {
                "status":
                    status,

                "helmet":
                    state[
                        "helmet"
                    ],

                "vest":
                    state[
                        "vest"
                    ],

                "issues":
                    issues,

                "overstay":
                    overstay,

                "dwell_sec":
                    dwell_sec
            }


        count = len(
            inside_persons
        )

        occupancy_alert = (
            self.update_occupancy(
                count,
                now
            )
        )


        ttl = float(
            self.timing[
                "track_ttl_sec"
            ]
        )

        expired = []

        for (
            track_id,
            state
        ) in self.tracks.items():

            if (
                now
                -
                state[
                    "last_seen"
                ]
                >
                ttl
            ):

                expired.append(
                    track_id
                )


        for track_id in expired:

            self.events.close_track(
                track_id,
                now
            )

            del self.tracks[
                track_id
            ]


        zone_alert = (
            any_person_warning
            or
            occupancy_alert
        )

        return (
            results,
            count,
            zone_alert
        )


# ============================================================
# OSD
# ============================================================

def yn(value):

    if value is True:
        return "Y"

    if value is False:
        return "N"

    return "?"


def set_person_label(
    obj_meta,
    result,
    case_cfg
):

    zone = case_cfg[
        "display_name"
    ]

    parts = [
        "ID {}".format(
            int(
                obj_meta.object_id
            )
        ),
        result[
            "status"
        ]
    ]

    if (
        case_cfg[
            "rules"
        ][
            "ppe"
        ][
            "enabled"
        ]
    ):

        parts.append(
            "H:{}".format(
                yn(
                    result[
                        "helmet"
                    ]
                )
            )
        )

        parts.append(
            "V:{}".format(
                yn(
                    result[
                        "vest"
                    ]
                )
            )
        )

    parts.append(
        zone
    )

    if (
        case_cfg[
            "rules"
        ][
            "dwell"
        ][
            "enabled"
        ]
    ):

        parts.append(
            "DWELL:{:.1f}s".format(
                result[
                    "dwell_sec"
                ]
            )
        )

    if result[
        "issues"
    ]:

        parts.extend(
            result[
                "issues"
            ]
        )

    if result[
        "overstay"
    ]:

        parts.append(
            "OVERSTAY"
        )

    text = " | ".join(
        parts
    )

    obj_meta.text_params.display_text = (
        text
    )

    obj_meta.text_params.font_params.font_size = 14

    obj_meta.text_params.font_params.font_color.set(
        1.0,
        1.0,
        1.0,
        1.0
    )

    obj_meta.text_params.set_bg_clr = 1

    obj_meta.text_params.text_bg_clr.set(
        0.0,
        0.0,
        0.0,
        0.65
    )

    if (
        result[
            "status"
        ]
        ==
        "WARNING"
    ):

        obj_meta.rect_params.border_color.set(
            1.0,
            0.1,
            0.1,
            1.0
        )

    elif (
        result[
            "status"
        ]
        ==
        "OBSERVING"
    ):

        obj_meta.rect_params.border_color.set(
            1.0,
            0.75,
            0.0,
            1.0
        )

    else:

        obj_meta.rect_params.border_color.set(
            0.1,
            1.0,
            0.2,
            1.0
        )

    obj_meta.rect_params.border_width = 3


def draw_zone(
    batch_meta,
    frame_meta,
    case_cfg,
    count,
    zone_alert
):

    display_meta = (
        pyds.nvds_acquire_display_meta_from_pool(
            batch_meta
        )
    )

    polygon = (
        case_cfg[
            "polygon_pixel"
        ]
    )

    color = (
        case_cfg[
            "visual"
        ][
            "border_rgba"
        ]
    )

    max_lines = min(
        len(polygon),
        16
    )

    display_meta.num_lines = (
        max_lines
    )

    for i in range(
        max_lines
    ):

        p1 = polygon[
            i
        ]

        p2 = polygon[
            (i + 1)
            %
            len(polygon)
        ]

        line = (
            display_meta.line_params[
                i
            ]
        )

        line.x1 = int(
            p1[0]
        )

        line.y1 = int(
            p1[1]
        )

        line.x2 = int(
            p2[0]
        )

        line.y2 = int(
            p2[1]
        )

        line.line_width = (
            6
            if
            zone_alert
            else
            4
        )

        if zone_alert:

            line.line_color.set(
                1.0,
                0.0,
                0.0,
                1.0
            )

        else:

            line.line_color.set(
                float(
                    color[0]
                ),
                float(
                    color[1]
                ),
                float(
                    color[2]
                ),
                float(
                    color[3]
                )
            )


    summary = (
        case_cfg[
            "display_name"
        ]
    )

    occupancy = (
        case_cfg[
            "rules"
        ][
            "occupancy"
        ]
    )

    dwell = (
        case_cfg[
            "rules"
        ][
            "dwell"
        ]
    )

    if occupancy[
        "enabled"
    ]:

        summary += (
            " | PERSONS:{}/{}".format(
                count,
                occupancy[
                    "max_people"
                ]
            )
        )

    else:

        summary += (
            " | PERSONS:{}".format(
                count
            )
        )

    if dwell[
        "enabled"
    ]:

        summary += (
            " | MAX DWELL:{:.0f}s".format(
                float(
                    dwell[
                        "limit_sec"
                    ]
                )
            )
        )

    if zone_alert:

        summary += " | ALERT"


    display_meta.num_labels = 1

    text = (
        display_meta.text_params[
            0
        ]
    )

    text.display_text = (
        summary
    )

    x = max(
        10,
        min(
            int(
                polygon[0][0]
            ),
            int(
                case_cfg[
                    "frame_width"
                ]
            )
            -
            600
        )
    )

    y = max(
        30,
        int(
            polygon[0][1]
        )
        -
        20
    )

    text.x_offset = x
    text.y_offset = y

    text.font_params.font_size = 18

    text.font_params.font_color.set(
        1.0,
        1.0,
        1.0,
        1.0
    )

    text.set_bg_clr = 1

    text.text_bg_clr.set(
        0.0,
        0.0,
        0.0,
        0.65
    )

    pyds.nvds_add_display_meta_to_frame(
        frame_meta,
        display_meta
    )


# ============================================================
# Application
# ============================================================

APP = None


def frame_time(
    frame_meta,
    fps
):

    pts = int(
        frame_meta.buf_pts
    )

    if (
        pts > 0
        and
        pts
        <
        10**18
    ):

        return (
            pts
            /
            1000000000.0
        )

    return (
        float(
            frame_meta.frame_num
        )
        /
        float(fps)
    )


def metadata_probe(
    pad,
    info,
    user_data
):

    global APP

    gst_buffer = (
        info.get_buffer()
    )

    if gst_buffer is None:

        return (
            Gst.PadProbeReturn.OK
        )


    batch_meta = (
        pyds.gst_buffer_get_nvds_batch_meta(
            hash(
                gst_buffer
            )
        )
    )

    if batch_meta is None:

        return (
            Gst.PadProbeReturn.OK
        )


    l_frame = (
        batch_meta.frame_meta_list
    )

    while l_frame is not None:

        frame_meta = (
            pyds.NvDsFrameMeta.cast(
                l_frame.data
            )
        )

        now = frame_time(
            frame_meta,
            APP.case_cfg[
                "fps"
            ]
        )

        persons = []
        helmets = []
        vests = []

        person_meta = {}


        l_obj = (
            frame_meta.obj_meta_list
        )

        while l_obj is not None:

            obj_meta = (
                pyds.NvDsObjectMeta.cast(
                    l_obj.data
                )
            )

            rect = (
                obj_meta.rect_params
            )

            item = {
                "class_id":
                    int(
                        obj_meta.class_id
                    ),

                "confidence":
                    float(
                        obj_meta.confidence
                    ),

                "track_id":
                    int(
                        obj_meta.object_id
                    ),

                "bbox": (
                    float(
                        rect.left
                    ),
                    float(
                        rect.top
                    ),
                    float(
                        rect.width
                    ),
                    float(
                        rect.height
                    )
                ),

                "meta":
                    obj_meta
            }


            if (
                item[
                    "class_id"
                ]
                ==
                PERSON
            ):

                persons.append(
                    item
                )

                if (
                    item[
                        "track_id"
                    ]
                    !=
                    UNTRACKED_ID
                ):

                    person_meta[
                        item[
                            "track_id"
                        ]
                    ] = (
                        obj_meta
                    )


            elif (
                item[
                    "class_id"
                ]
                ==
                HELMET
            ):

                helmets.append(
                    item
                )


            elif (
                item[
                    "class_id"
                ]
                ==
                VEST
            ):

                vests.append(
                    item
                )


            try:

                l_obj = (
                    l_obj.next
                )

            except StopIteration:

                break


        (
            results,
            count,
            zone_alert
        ) = APP.engine.process(
            persons,
            helmets,
            vests,
            now
        )


        # Only persons confirmed inside the active ROI
        # receive custom PERL labels.
        # Persons outside the ROI retain normal YOLO/DeepStream labels.
        for (
            track_id,
            result
        ) in results.items():

            meta = (
                person_meta.get(
                    track_id
                )
            )

            if meta is not None:

                set_person_label(
                    meta,
                    result,
                    APP.case_cfg
                )


        draw_zone(
            batch_meta,
            frame_meta,
            APP.case_cfg,
            count,
            zone_alert
        )

        APP.frame_count += 1


        try:

            l_frame = (
                l_frame.next
            )

        except StopIteration:

            break


    return (
        Gst.PadProbeReturn.OK
    )


def decodebin_pad_added(
    decodebin,
    src_pad,
    source_bin
):

    caps = (
        src_pad.get_current_caps()
    )

    if caps is None:

        caps = (
            src_pad.query_caps(
                None
            )
        )

    structure = (
        caps.get_structure(
            0
        )
    )

    name = (
        structure.get_name()
    )

    if not name.startswith(
        "video"
    ):

        return

    features = (
        caps.get_features(
            0
        )
    )

    if not features.contains(
        "memory:NVMM"
    ):

        print(
            "SOURCE_NVMM_STATUS=FAIL"
        )

        return

    ghost = (
        source_bin.get_static_pad(
            "src"
        )
    )

    if not ghost.set_target(
        src_pad
    ):

        print(
            "SOURCE_GHOST_LINK=FAIL"
        )


def create_source_bin(
    uri
):

    source_bin = (
        Gst.Bin.new(
            "source-bin"
        )
    )

    decode = (
        Gst.ElementFactory.make(
            "uridecodebin",
            "uri-decode-bin"
        )
    )

    if (
        source_bin is None
        or
        decode is None
    ):

        raise RuntimeError(
            "Unable to create source"
        )

    decode.set_property(
        "uri",
        uri
    )

    decode.connect(
        "pad-added",
        decodebin_pad_added,
        source_bin
    )

    source_bin.add(
        decode
    )

    ghost = (
        Gst.GhostPad.new_no_target(
            "src",
            Gst.PadDirection.SRC
        )
    )

    source_bin.add_pad(
        ghost
    )

    return source_bin


class App:

    def __init__(
        self,
        args
    ):

        self.args = args

        with open(
            args.case_config,
            "r"
        ) as f:

            self.case_cfg = (
                json.load(
                    f
                )
            )

        self.events = (
            EventWriter(
                args.event_dir
            )
        )

        self.engine = (
            PERLEngine(
                self.case_cfg,
                self.events
            )
        )

        self.pipeline = None

        self.loop = None

        self.frame_count = 0


    def build(
        self
    ):

        Gst.init(None)

        pipeline = (
            Gst.Pipeline.new(
                "perl-runtime"
            )
        )

        streammux = (
            Gst.ElementFactory.make(
                "nvstreammux",
                "streammux"
            )
        )

        pgie = (
            Gst.ElementFactory.make(
                "nvinfer",
                "pgie"
            )
        )

        tracker = (
            Gst.ElementFactory.make(
                "nvtracker",
                "tracker"
            )
        )

        convert = (
            Gst.ElementFactory.make(
                "nvvideoconvert",
                "convert"
            )
        )

        osd = (
            Gst.ElementFactory.make(
                "nvdsosd",
                "osd"
            )
        )

        sink = (
            Gst.ElementFactory.make(
                "nveglglessink",
                "sink"
            )
        )

        if any(
            x is None
            for x
            in [
                pipeline,
                streammux,
                pgie,
                tracker,
                convert,
                osd,
                sink
            ]
        ):

            raise RuntimeError(
                "Unable to create DeepStream pipeline elements"
            )


        video_path = (
            self.case_cfg[
                "video_path"
            ]
        )

        if not os.path.isfile(
            video_path
        ):

            raise FileNotFoundError(
                video_path
            )


        uri = (
            "file://"
            +
            video_path
        )

        source = (
            create_source_bin(
                uri
            )
        )


        pipeline.add(
            source
        )

        for element in [
            streammux,
            pgie,
            tracker,
            convert,
            osd
        ]:

            pipeline.add(
                element
            )


        streammux.set_property(
            "width",
            int(
                self.case_cfg[
                    "frame_width"
                ]
            )
        )

        streammux.set_property(
            "height",
            int(
                self.case_cfg[
                    "frame_height"
                ]
            )
        )

        streammux.set_property(
            "batch-size",
            1
        )

        streammux.set_property(
            "live-source",
            0
        )

        streammux.set_property(
            "batched-push-timeout",
            40000
        )


        pgie.set_property(
            "config-file-path",
            self.args.infer_config
        )


        tracker.set_property(
            "ll-lib-file",
            self.args.tracker_lib
        )

        tracker.set_property(
            "ll-config-file",
            self.args.tracker_config
        )

        tracker.set_property(
            "enable-batch-process",
            1
        )

        tracker.set_property(
            "display-tracking-id",
            1
        )

        tracker.set_property(
            "tracker-width",
            640
        )

        tracker.set_property(
            "tracker-height",
            384
        )


        sink.set_property(
            "sync",
            0
        )

        sink.set_property(
            "qos",
            0
        )


        source_src = (
            source.get_static_pad(
                "src"
            )
        )

        mux_sink = (
            streammux.get_request_pad(
                "sink_0"
            )
        )

        if (
            source_src.link(
                mux_sink
            )
            !=
            Gst.PadLinkReturn.OK
        ):

            raise RuntimeError(
                "source -> streammux failed"
            )


        if not streammux.link(
            pgie
        ):

            raise RuntimeError(
                "streammux -> pgie failed"
            )

        if not pgie.link(
            tracker
        ):

            raise RuntimeError(
                "pgie -> tracker failed"
            )

        if not tracker.link(
            convert
        ):

            raise RuntimeError(
                "tracker -> convert failed"
            )

        if not convert.link(
            osd
        ):

            raise RuntimeError(
                "convert -> osd failed"
            )


        if (
            platform.machine()
            ==
            "aarch64"
        ):

            transform = (
                Gst.ElementFactory.make(
                    "nvegltransform",
                    "egl-transform"
                )
            )

            if transform is None:

                raise RuntimeError(
                    "Unable to create nvegltransform"
                )

            pipeline.add(
                transform
            )

            pipeline.add(
                sink
            )

            if not osd.link(
                transform
            ):

                raise RuntimeError(
                    "osd -> transform failed"
                )

            if not transform.link(
                sink
            ):

                raise RuntimeError(
                    "transform -> sink failed"
                )

        else:

            pipeline.add(
                sink
            )

            if not osd.link(
                sink
            ):

                raise RuntimeError(
                    "osd -> sink failed"
                )


        probe_pad = (
            tracker.get_static_pad(
                "src"
            )
        )

        probe_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            metadata_probe,
            None
        )


        self.pipeline = (
            pipeline
        )


    def on_bus(
        self,
        bus,
        message
    ):

        if (
            message.type
            ==
            Gst.MessageType.EOS
        ):

            print(
                "PERL_EOS=PASS"
            )

            self.loop.quit()


        elif (
            message.type
            ==
            Gst.MessageType.ERROR
        ):

            err, debug = (
                message.parse_error()
            )

            print(
                "PERL_GST_ERROR=",
                err
            )

            print(
                "PERL_GST_DEBUG=",
                debug
            )

            self.loop.quit()


        return True


    def run(
        self
    ):

        self.build()

        self.loop = (
            GLib.MainLoop()
        )

        bus = (
            self.pipeline.get_bus()
        )

        bus.add_signal_watch()

        bus.connect(
            "message",
            self.on_bus
        )


        print(
            "CASE_ID=",
            self.case_cfg[
                "case_id"
            ]
        )

        print(
            "ZONE=",
            self.case_cfg[
                "display_name"
            ]
        )

        print(
            "VIDEO=",
            self.case_cfg[
                "video_path"
            ]
        )

        print(
            "EVENT_DIR=",
            self.args.event_dir
        )

        print(
            "PERL_START=PASS"
        )


        self.pipeline.set_state(
            Gst.State.PLAYING
        )

        try:

            self.loop.run()

        finally:

            self.events.shutdown(
                self.engine.last_time
            )

            self.pipeline.set_state(
                Gst.State.NULL
            )

            print(
                "FRAME_COUNT=",
                self.frame_count
            )

            print(
                "PERL_STOP=PASS"
            )


def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--case-config",
        required=True
    )

    parser.add_argument(
        "--infer-config",
        required=True
    )

    parser.add_argument(
        "--event-dir",
        required=True
    )

    parser.add_argument(
        "--tracker-lib",
        required=True
    )

    parser.add_argument(
        "--tracker-config",
        required=True
    )

    return parser.parse_args()


def main():

    global APP

    args = (
        parse_args()
    )

    try:

        APP = App(
            args
        )

        APP.run()

        print(
            "PERL_EXIT=0"
        )

        return 0

    except Exception as e:

        print(
            "PERL_FATAL=",
            repr(e)
        )

        traceback.print_exc()

        print(
            "PERL_EXIT=1"
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
