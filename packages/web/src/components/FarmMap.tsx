"use client";

import { useRouter } from "next/navigation";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useI18n, localizedTreeName } from "@/lib/i18n";
import { STATUS_COLOR, STATUS_ORDER } from "@/lib/status";
import type { Tree } from "@/lib/types";

function center(trees: Tree[]): [number, number] {
  if (trees.length === 0) return [25.383, 49.586]; // Al-Ahsa default
  const lat = trees.reduce((s, t) => s + t.lat, 0) / trees.length;
  const lon = trees.reduce((s, t) => s + t.lon, 0) / trees.length;
  return [lat, lon];
}

export default function FarmMap({ trees }: { trees: Tree[] }) {
  const { t, locale } = useI18n();
  const router = useRouter();

  return (
    <>
      <div className="legend">
        {STATUS_ORDER.map((status) => (
          <span className="legend-item" key={status}>
            <span className="dot" style={{ background: STATUS_COLOR[status] }} />
            {t(`status.${status}`)}
          </span>
        ))}
      </div>
      <div className="map">
        <MapContainer center={center(trees)} zoom={16} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {trees.map((tree) => (
            <CircleMarker
              key={tree.id}
              center={[tree.lat, tree.lon]}
              radius={10}
              pathOptions={{
                color: STATUS_COLOR[tree.status],
                fillColor: STATUS_COLOR[tree.status],
                fillOpacity: 0.85,
              }}
              eventHandlers={{ click: () => router.push(`/tree/${tree.id}`) }}
            >
              <Popup>
                <strong>{localizedTreeName(locale, tree)}</strong>
                <br />
                {t(`status.${tree.status}`)}
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </>
  );
}
