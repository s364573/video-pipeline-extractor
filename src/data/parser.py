def get_events(df, event_name):
    return df[df["event"] == event_name]


def get_event_pairs(df, start_event, end_event):
    starts = df[df["event"] == start_event].reset_index(drop=True)
    ends   = df[df["event"] == end_event].reset_index(drop=True)

    assert len(starts) == len(ends), "Mismatch start/end events"

    pairs = []
    for i in range(len(starts)):
        pairs.append({
            "start": starts.loc[i, "t"],
            "end": ends.loc[i, "t"],
            "detail": starts.loc[i, "detail"]
        })

    return pairs