import { useMemo } from "react";
import { CalendarDays, Moon, Sun, Sunrise, Sunset, Info } from "lucide-react";
import {
  type Coordinate, type LocalInstant, solarDay, solarElevation, lightPhase,
  localParts, formatLocal, utcOffsetLabel,
} from "./local-time";

type Props = {
  date: string;
  hour: string;
  zone: string;
  centre: Coordinate;
  candidates: LocalInstant[];
  selected: LocalInstant | undefined;
  disabled: boolean;
  hasArea: boolean;
  onDate: (date: string) => void;
  onHour: (hour: string) => void;
  onInstant: (utc: string) => void;
  zoneChoice?: { value: string; onChange: (zone: string) => void };
  inputIdPrefix?: string;
  referenceLabel?: string;
  supportDescription?: string;
  minDate?: string;
  maxDate?: string;
};
const colors = { day: "#f7c86a", twilight: "#94779e", night: "#29405c" };
const labels = { day: "Daylight", twilight: "Twilight", night: "Night" };
const yFor = (elevation: number) => Math.max(10, Math.min(85, 59 - elevation * .68));

export function LocalTimePicker(props: Props) {
  const { date, hour, zone, centre, candidates, selected, disabled } = props;
  const id = props.inputIdPrefix || "";
  const timeline = useMemo(() => solarDay(date, zone, centre), [date, zone, centre[0], centre[1]]);
  const stamp = selected ? Date.parse(selected.utc) : null;
  const elevation = stamp !== null ? solarElevation(stamp, centre) : null;
  const phase = elevation === null ? null : lightPhase(elevation);
  const position = timeline && stamp !== null
    ? Math.max(0, Math.min(1, (stamp - timeline.start) / (timeline.end - timeline.start))) : null;
  const gradient = timeline ? `linear-gradient(to right, ${timeline.samples.map((sample, i) => {
    const next = timeline.samples[i + 1]?.position ?? 1;
    return `${colors[sample.phase]} ${sample.position * 100}% ${next * 100}%`;
  }).join(", ")})` : "none";
  const path = timeline?.samples.map((sample, i) => `${i ? "L" : "M"}${(sample.position * 300).toFixed(2)},${yFor(sample.elevation).toFixed(2)}`).join(" ");
  return (
    <div className="local-time-picker">
      <div className="local-zone"><CalendarDays size={13} /><span>Local time <strong>{zone.replaceAll("_", " ")}</strong></span></div>
      {props.zoneChoice && (
        <label className="zone-choice">Clock convention
          <select value={props.zoneChoice.value} onChange={(e) => props.zoneChoice!.onChange(e.target.value)} disabled={disabled}>
            <option value="Asia/Singapore">Singapore</option>
            <option value="Asia/Kuala_Lumpur">Johor · peninsular Malaysia</option>
          </select>
          <small>These clocks agree since 1970; choose the area for older dates.</small>
        </label>
      )}
      <div className="date-row">
        <div>
          <label htmlFor={`${id}date`}>Local date</label>
          <input id={`${id}date`} type="date" min={props.minDate || "1900-01-01"} max={props.maxDate || "2100-12-31"} value={date}
            onInput={(e) => props.onDate(e.currentTarget.value)}
            onChange={(e) => props.onDate(e.target.value)} disabled={disabled} />
        </div>
        <div>
          <label htmlFor={`${id}hour`}>Local time</label>
          <input id={`${id}hour`} type="time" step="60" value={hour}
            onInput={(e) => props.onHour(e.currentTarget.value)}
            onChange={(e) => props.onHour(e.target.value)} disabled={disabled} />
        </div>
      </div>
      {date && hour && !candidates.length && (
        <div className="time-resolution-error" role="alert"><Info size={14} /><span>
          {timeline ? "This local time does not exist because the clocks move forward. Choose another time; nothing has been shifted automatically."
            : "Enter a valid local date and time."}
        </span></div>
      )}
      {candidates.length > 1 && (
        <fieldset className="time-fold" disabled={disabled}>
          <legend>This clock time occurs twice. Choose one.</legend>
          {candidates.map((candidate, i) => <label key={candidate.utc}>
            <input type="radio" name={`${id}time-occurrence`} checked={selected?.utc === candidate.utc} onChange={() => props.onInstant(candidate.utc)} />
            <span>{i === 0 ? "First" : "Second"} · {utcOffsetLabel(candidate.offsetSeconds)}<small>{candidate.utc.replace("T", " ").replace(".000Z", " UTC")}</small></span>
          </label>)}
        </fieldset>
      )}
      {timeline && (
        <div className="solar-clock">
          <div className="solar-clock-heading">
            <span>{phase ? labels[phase] : "Choose a moment"}</span>
            <small>{props.hasArea ? "At your area’s centre" : props.referenceLabel || "At the pilot centre"}</small>
          </div>
          <div className="solar-arc" aria-hidden="true">
            <svg viewBox="0 0 300 94" preserveAspectRatio="none">
              {timeline.samples.slice(0, -1).map((sample, i) => <rect key={i} x={sample.position * 300} y="0" width={(timeline.samples[i + 1].position - sample.position) * 300 + .1} height="94" fill={colors[sample.phase]} opacity=".12" />)}
              <line x1="0" x2="300" y1="59" y2="59" stroke="#8196a4" strokeOpacity=".45" strokeDasharray="3 4" />
              <path d={path} fill="none" stroke="#d4c4a6" strokeWidth="1.5" opacity=".7" />
              {timeline.events.map((event) => <line key={event.kind} x1={event.position * 300} x2={event.position * 300} y1="48" y2="94" stroke="#d6c4a7" strokeOpacity=".55" strokeDasharray="2 3" />)}
              {position !== null && <line x1={position * 300} x2={position * 300} y1="0" y2="94" stroke="#c3fff2" strokeOpacity=".35" />}
            </svg>
            {position !== null && elevation !== null && <div className={`solar-body ${phase}`} style={{ left: `${position * 100}%`, top: `${yFor(elevation) / 94 * 100}%` }}>
              {phase === "day" ? <Sun size={19} /> : <Moon size={17} />}
            </div>}
            <span className="horizon-label">Horizon</span>
          </div>
          <div className="solar-slider" style={{ background: gradient }}>
            <input aria-label="Local time across the selected day" type="range" min="0"
              max={Math.max(0, Math.floor((timeline.end - timeline.start) / 60000) - 1)} step="1"
              value={position === null ? 0 : Math.round((stamp! - timeline.start) / 60000)}
              aria-valuetext={selected ? `${formatLocal(selected.utc, zone)}, ${phase ? labels[phase] : ""}` : "No unambiguous time selected"}
              onChange={(e) => props.onInstant(new Date(timeline.start + Number(e.target.value) * 60000).toISOString())}
              disabled={disabled} />
          </div>
          <div className="solar-ticks" aria-hidden="true">{timeline.ticks.map((tick, i) => <span key={i} style={{ left: `${tick.position * 100}%` }}>{tick.label}</span>)}</div>
          <div className="solar-phase-key"><span><i className="night" />Night</span><span><i className="twilight" />Twilight</span><span><i className="day" />Daylight</span></div>
          <div className="solar-events">
            {timeline.events.length ? timeline.events.map((event) => <span key={event.kind}>
              {event.kind === "sunrise" ? <Sunrise size={14} /> : <Sunset size={14} />}
              {event.kind === "sunrise" ? "Sunrise" : "Sunset"}<strong>≈ {localParts(event.stamp, zone).time}</strong>
            </span>) : <span>{timeline.samples.every((s) => s.elevation > -.833)
              ? "Sun stays above the horizon on this date."
              : timeline.samples.every((s) => s.elevation <= -.833)
                ? "Sun stays below the horizon on this date."
                : "No sunrise or sunset falls within this local date."}</span>}
          </div>
          {timeline.durationHours !== 24 && <p className="clock-change-note">Clocks change today · {timeline.durationHours} hours. The slider follows actual elapsed time.</p>}
        </div>
      )}
      <div className="canonical-time" aria-live="polite">
        <div><span>Local</span><strong>{selected ? formatLocal(selected.utc, zone) : "Choose an unambiguous time"}</strong></div>
        <div><span>Request UTC</span><strong>{selected ? selected.utc.replace("T", " ").replace(".000Z", " UTC") : "—"}</strong></div>
      </div>
      <p className="helper small solar-guide-note">{props.supportDescription || "Sun times are approximate and ignore terrain and cast shadows. Twilight means the sun is up to 6° below the horizon. The server checks daylight across your full selection."}</p>
    </div>
  );
}
