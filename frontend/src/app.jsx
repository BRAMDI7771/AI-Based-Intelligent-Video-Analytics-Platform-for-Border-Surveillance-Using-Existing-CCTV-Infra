import { useEffect, useMemo, useState } from "react";
const BACKEND = "https://safer-improvement-recognize-mode.trycloudflare.com";

function App() {
  const [running, setRunning] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);

  const [stats, setStats] = useState({
    personsCount: 0,
    vehiclesCount: 0,
    fps: 0,
    recognizedName: "UNKNOWN",
    recognitionConfidence: 0,
    vehicleName: "NONE",
    status: "SYSTEM IDLE",
  });

  const [error, setError] = useState("");
  const [loadingCameras, setLoadingCameras] = useState(false);

  // ---------------------------------------------------------
  // CAMERA LIST
  // ---------------------------------------------------------

  async function loadCameras() {
  try {
    setLoadingCameras(true);

    const response = await fetch(
      `${BACKEND}/api/cameras`,
      {
        cache: "no-store",
      }
    );

    if (!response.ok) {
      throw new Error(
        `Camera API error: ${response.status}`
      );
    }

    const data = await response.json();

    const cameraList = Array.isArray(data.cameras)
      ? data.cameras
      : [];

    setCameras(cameraList);

    // --------------------------------------------------
    // SELECT FIRST CAMERA
    // --------------------------------------------------

    if (cameraList.length > 0) {
      setSelectedCameraId((current) => {
        const exists = cameraList.some(
          (camera) =>
            camera.camera_id === current
        );

        return exists
          ? current
          : cameraList[0].camera_id;
      });
    } else {
      setSelectedCameraId(null);
    }

    // --------------------------------------------------
    // DASHBOARD STATS
    // --------------------------------------------------

    let totalPersons = 0;
    let totalVehicles = 0;
    let totalFps = 0;
    let fpsCameras = 0;

    let bestName = "UNKNOWN";
    let bestSimilarity = 0;
    let bestVehicle = "NONE";

    let hasLiveCamera = false;

    for (const camera of cameraList) {
      totalPersons += Number(
        camera.persons ?? 0
      );

      totalVehicles += Number(
        camera.vehicles ?? 0
      );

      const fps = Number(
        camera.fps ?? 0
      );

      if (fps > 0) {
        totalFps += fps;
        fpsCameras += 1;
      }

      if (
        camera.active &&
        camera.video_received
      ) {
        hasLiveCamera = true;
      }

      const name =
        camera.display_name;

      const similarity = Number(
        camera.similarity ?? 0
      );

      if (
        name &&
        name !== "UNKNOWN" &&
        name !== "UNKNOWN PERSON" &&
        similarity > bestSimilarity
      ) {
        bestName = name;
        bestSimilarity = similarity;
      }

      if (
        camera.vehicle_name &&
        camera.vehicle_name !== "NONE"
      ) {
        bestVehicle =
          camera.vehicle_name;
      }
    }

    setStats({
      personsCount: totalPersons,
      vehiclesCount: totalVehicles,

      fps:
        fpsCameras > 0
          ? totalFps / fpsCameras
          : 0,

      recognizedName: bestName,

      recognitionConfidence:
        bestSimilarity,

      vehicleName: bestVehicle,

      status: hasLiveCamera
        ? "SYSTEM MONITORING"
        : "SYSTEM IDLE",
    });

  } catch (err) {
    console.error(
      "Camera list error:",
      err
    );
  } finally {
    setLoadingCameras(false);
  }
}
  // ---------------------------------------------------------
  // START CENTRALIZED LAPTOP CAMERA
  // ---------------------------------------------------------

  async function startSurveillance() {
  try {
    setError("");
    setRunning(true);

    const response = await fetch(`${BACKEND}/api/laptop/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.message || "Unable to start laptop camera");
    }

    console.log("Laptop camera:", data.message);

    setStats((prev) => ({
      ...prev,
      status: "SYSTEM MONITORING",
    }));

  } catch (err) {
    console.error("Start surveillance error:", err);

    setError(err.message || "Unable to start surveillance");
    setRunning(false);
  }
}

  // ---------------------------------------------------------
  // STOP CENTRALIZED LAPTOP CAMERA
  // ---------------------------------------------------------

  async function stopSurveillance() {
  try {
    await fetch(`${BACKEND}/api/laptop/stop`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
  } catch (err) {
    console.error("Stop camera error:", err);
  }

  setRunning(false);

  setStats({
    personsCount: 0,
    vehiclesCount: 0,
    fps: 0,
    recognizedName: "UNKNOWN",
    recognitionConfidence: 0,
    vehicleName: "NONE",
    status: "SYSTEM IDLE",
  });
}
// ---------------------------------------------------------
// REMOTE CAMERA
// ---------------------------------------------------------

function openRemoteCameraPage() {
  window.open(
    `${BACKEND}/phone`,
    "_blank",
    "noopener,noreferrer"
  );
}
  // ---------------------------------------------------------
  // POLLING
  // ---------------------------------------------------------

  useEffect(() => {
  loadCameras();

  const timer = setInterval(() => {
    loadCameras();
  }, 2000);

  return () => {
    clearInterval(timer);
    
  };
}, []);

  // ---------------------------------------------------------
  // CLEANUP ON PAGE EXIT
  // ---------------------------------------------------------

  

  // ---------------------------------------------------------
  // SELECTED CAMERA
  // ---------------------------------------------------------

  const selectedCamera = useMemo(() => {
    return cameras.find(
      (camera) =>
        camera.camera_id ===
        selectedCameraId
    );
  }, [
    cameras,
    selectedCameraId,
  ]);

  const confidencePercent =
    Number(
      stats.recognitionConfidence || 0
    ) * 100;

  // ---------------------------------------------------------
  // CAMERA FEED URL
  // ---------------------------------------------------------

  function getSnapshotUrl(cameraId) {
    return `${BACKEND}/snapshot/${encodeURIComponent(
      cameraId
    )}`;
  }

  function getMjpegUrl(cameraId) {
    return `${BACKEND}/mjpeg/${encodeURIComponent(
      cameraId
    )}`;
  }

  // ---------------------------------------------------------
  // CAMERA CARD
  // ---------------------------------------------------------

  function CameraCard({ camera }) {
    const cameraId = camera.camera_id;

    const isLive =
      Boolean(camera.video_received) ||
      camera.active === true ||
      camera.connection_state ===
        "connected";

    const fps = Number(
      camera.fps ?? 0
    );

    return (
      <div
        onClick={() =>
          setSelectedCameraId(cameraId)
        }
        style={{
          background: "#111827",
          border: "1px solid #243244",
          borderRadius: "14px",
          overflow: "hidden",
          cursor: "pointer",
          boxShadow:
            selectedCameraId === cameraId
              ? "0 0 0 2px #38bdf8"
              : "0 8px 30px rgba(0,0,0,0.18)",
        }}
      >
        {/* CAMERA HEADER */}
        <div
          style={{
            padding:
              "14px 16px 12px 16px",
            display: "flex",
            justifyContent:
              "space-between",
            alignItems: "center",
            background:
              "#0f172a",
          }}
        >
          <div>
            <div
              style={{
                fontSize: "18px",
                fontWeight: 800,
                color: "#f8fafc",
              }}
            >
              {cameraId}
            </div>

            <div
              style={{
                marginTop: "3px",
                fontSize: "12px",
                color: "#94a3b8",
                textTransform:
                  "uppercase",
              }}
            >
              {camera.source_type ||
                "REMOTE CAMERA"}
            </div>
          </div>

          <div
            style={{
              color: isLive
                ? "#22c55e"
                : "#f59e0b",
              fontWeight: 800,
              fontSize: "13px",
            }}
          >
            ● {isLive
              ? "LIVE"
              : "CONNECTING"}
          </div>
        </div>

        {/* LIVE CAMERA */}
        <div
          style={{
            width: "100%",
            height: "360px",
            background: "#020617",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            overflow: "hidden",
          }}
        >
          <img
            src={getMjpegUrl(cameraId)}
            alt={`${cameraId} surveillance feed`}
            loading="eager"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
          />
        </div>

        {/* CAMERA INFO */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(4, minmax(0,1fr))",
            gap: "8px",
            padding: "12px",
            background:
              "#0b1220",
          }}
        >
          <InfoBox
            title="FPS"
            value={fps.toFixed(1)}
          />

          <InfoBox
            title="EVENTS"
            value={
              camera.events ?? 0
            }
          />

          <InfoBox
            title="FRAMES"
            value={
              camera.processed_frames ??
              0
            }
          />

          <InfoBox
            title="STATUS"
            value={
              camera.connection_state ||
              (isLive
                ? "LIVE"
                : "OFF")
            }
          />
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg,#07101f 0%,#0b1120 100%)",
        color: "#fff",
        padding: "24px",
        fontFamily:
          "Arial, Helvetica, sans-serif",
      }}
    >
      {/* =====================================================
          HEADER
      ====================================================== */}

      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems: "center",
          gap: "20px",
          marginBottom: "22px",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h1
            style={{
              margin: 0,
              fontSize: "34px",
              fontWeight: 900,
              letterSpacing:
                "0.5px",
            }}
          >
            BORDER SURVEILLANCE AI
          </h1>

          <p
            style={{
              margin:
                "8px 0 0 0",
              color: "#94a3b8",
              fontSize: "15px",
            }}
          >
            AI Powered CCTV Monitoring
            & Face Recognition
          </p>
        </div>

        <div
          style={{
            display: "flex",
            gap: "10px",
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <div
            style={{
              padding:
                "11px 16px",
              borderRadius: "10px",
              background: running
                ? "#14532d"
                : "#334155",
              color: running
                ? "#86efac"
                : "#e2e8f0",
              fontWeight: 800,
              whiteSpace:
                "nowrap",
            }}
          >
            ● {stats.status}
          </div>

          {!running ? (
            <button
              onClick={
                startSurveillance
              }
              style={{
                border: "none",
                borderRadius: "10px",
                padding:
                  "13px 20px",
                background:
                  "#16a34a",
                color: "white",
                fontWeight: 900,
                fontSize: "14px",
                cursor: "pointer",
              }}
            >
              ▶ START SURVEILLANCE
            </button>
          ) : (
            <button
              onClick={
                stopSurveillance
              }
              style={{
                border: "none",
                borderRadius: "10px",
                padding:
                  "13px 20px",
                background:
                  "#dc2626",
                color: "white",
                fontWeight: 900,
                fontSize: "14px",
                cursor: "pointer",
              }}
            >
              ■ STOP SURVEILLANCE
            </button>
          )}

          <button
            onClick={
              openRemoteCameraPage
            }
            style={{
              border:
                "1px solid #334155",
              borderRadius: "10px",
              padding:
                "13px 18px",
              background:
                "#111827",
              color: "#cbd5e1",
              fontWeight: 800,
              fontSize: "14px",
              cursor: "pointer",
            }}
          >
            + CONNECT REMOTE CAMERA
          </button>
        </div>
      </div>

      {/* =====================================================
          ERROR
      ====================================================== */}

      {error && (
        <div
          style={{
            background:
              "#7f1d1d",
            border:
              "1px solid #b91c1c",
            color: "#fecaca",
            padding: "13px 16px",
            borderRadius: "10px",
            marginBottom: "18px",
            fontWeight: 700,
          }}
        >
          {error}
        </div>
      )}

      {/* =====================================================
          TOP STATS
      ====================================================== */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(4,minmax(0,1fr))",
          gap: "14px",
          marginBottom: "18px",
        }}
      >
        <StatCard
          title="PERSONS"
          value={
            stats.personsCount
          }
          accent="#60a5fa"
        />

        <StatCard
          title="VEHICLES"
          value={
            stats.vehiclesCount
          }
          accent="#38bdf8"
        />

        <StatCard
          title="FPS"
          value={Number(
            stats.fps || 0
          ).toFixed(1)}
          accent="#a78bfa"
        />

        <StatCard
          title="ACTIVE CAMERAS"
          value={cameras.length}
          accent="#22c55e"
        />
      </div>

      {/* =====================================================
          AI INFORMATION
      ====================================================== */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2,minmax(0,1fr))",
          gap: "16px",
          marginBottom: "22px",
        }}
      >
        {/* FACE RECOGNITION */}

        <div
          style={{
            background:
              "linear-gradient(135deg,#101b35,#15254a)",
            border:
              "1px solid #2d4777",
            borderRadius: "12px",
            padding: "18px",
          }}
        >
          <h2
            style={{
              margin:
                "0 0 16px 0",
              color: "#60a5fa",
              fontSize: "21px",
            }}
          >
            FACE RECOGNITION
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(3,minmax(0,1fr))",
              gap: "14px",
            }}
          >
            <Metric
              label="IDENTIFIED PERSON"
              value={
                stats.recognizedName ||
                "UNKNOWN"
              }
            />

            <Metric
              label="MATCH SIMILARITY"
              value={Number(
                stats.recognitionConfidence ||
                  0
              ).toFixed(3)}
            />

            <Metric
              label="CONFIDENCE"
              value={`${confidencePercent.toFixed(
                1
              )}%`}
            />
          </div>
        </div>

        {/* VEHICLE IDENTIFICATION */}

        <div
          style={{
            background:
              "linear-gradient(135deg,#101b35,#15254a)",
            border:
              "1px solid #2d4777",
            borderRadius: "12px",
            padding: "18px",
          }}
        >
          <h2
            style={{
              margin:
                "0 0 16px 0",
              color: "#22d3ee",
              fontSize: "21px",
            }}
          >
            VEHICLE IDENTIFICATION
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(3,minmax(0,1fr))",
              gap: "14px",
            }}
          >
            <Metric
              label="VEHICLES DETECTED"
              value={
                stats.vehiclesCount
              }
            />

            <Metric
              label="VEHICLE TYPE"
              value={
                stats.vehicleName ||
                "NONE"
              }
            />

            <Metric
              label="ANPR STATUS"
              value="OCR ACTIVE / PENDING"
            />
          </div>
        </div>
      </div>

      {/* =====================================================
          CENTRALIZED CAMERA MONITORING
      ====================================================== */}

      <div
        style={{
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent:
              "space-between",
            alignItems: "center",
            gap: "15px",
            marginBottom: "14px",
            flexWrap: "wrap",
          }}
        >
          <h2
            style={{
              margin: 0,
              color: "#60a5fa",
              fontSize: "25px",
            }}
          >
            CENTRALIZED CAMERA
            MONITORING
          </h2>

          <div
            style={{
              color: "#22c55e",
              fontWeight: 900,
              fontSize: "14px",
            }}
          >
            ● {cameras.length}
            {" CAMERA(S) CONNECTED"}
          </div>
        </div>

        {loadingCameras &&
          cameras.length === 0 && (
            <div
              style={{
                padding: "30px",
                borderRadius: "12px",
                background:
                  "#111827",
                color: "#94a3b8",
                textAlign:
                  "center",
              }}
            >
              CONNECTING TO CENTRAL
              CAMERA SYSTEM...
            </div>
          )}

        {!loadingCameras &&
          cameras.length === 0 && (
            <div
              style={{
                padding: "40px",
                border:
                  "1px dashed #334155",
                borderRadius: "12px",
                background:
                  "#0f172a",
                color: "#94a3b8",
                textAlign:
                  "center",
              }}
            >
              <div
                style={{
                  fontSize: "18px",
                  fontWeight: 800,
                  color: "#cbd5e1",
                }}
              >
                NO CAMERAS CONNECTED
              </div>

              <div
                style={{
                  marginTop: "8px",
                  fontSize: "14px",
                }}
              >
                Start the laptop camera
                or connect a remote
                camera.
              </div>
            </div>
          )}

        {cameras.length > 0 && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(2,minmax(0,1fr))",
              gap: "18px",
            }}
          >
            {cameras.map(
              (camera) => (
                <CameraCard
                  key={
                    camera.camera_id
                  }
                  camera={camera}
                />
              )
            )}
          </div>
        )}
      </div>

      {/* =====================================================
          LARGE AI SURVEILLANCE FEED
      ====================================================== */}

      <div
        style={{
          background:
            "#0f172a",
          border:
            "1px solid #25354d",
          borderRadius: "14px",
          overflow: "hidden",
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            padding:
              "15px 18px",
            display: "flex",
            justifyContent:
              "space-between",
            alignItems: "center",
            gap: "15px",
            flexWrap: "wrap",
            background:
              "#111c2f",
          }}
        >
          <div>
            <h2
              style={{
                margin: 0,
                color: "#38bdf8",
                fontSize: "24px",
              }}
            >
              AI SURVEILLANCE FEED
            </h2>

            <div
              style={{
                marginTop: "5px",
                color: "#94a3b8",
                fontSize: "13px",
              }}
            >
              Real-time processed camera
              output
            </div>
          </div>

          {selectedCamera && (
            <div
              style={{
                padding:
                  "8px 12px",
                borderRadius: "8px",
                background:
                  "#172554",
                color: "#93c5fd",
                fontWeight: 800,
                fontSize: "13px",
              }}
            >
              CAMERA:{" "}
              {
                selectedCamera.camera_id
              }
            </div>
          )}
        </div>

        {selectedCamera ? (
          <div
            style={{
              width: "100%",
              height: "560px",
              background:
                "#020617",
            }}
          >
            <img
              src={getMjpegUrl(
                selectedCamera.camera_id
              )}
              alt="AI processed surveillance feed"
              loading="eager"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                display: "block",
                background:
                  "#020617",
              }}
            />
          </div>
        ) : (
          <div
            style={{
              height: "420px",
              display: "flex",
              alignItems: "center",
              justifyContent:
                "center",
              background:
                "#020617",
              color: "#64748b",
              fontSize: "17px",
              fontWeight: 700,
            }}
          >
            AI FEED WILL APPEAR
            HERE WHEN A CAMERA
            CONNECTS
          </div>
        )}
      </div>

      {/* =====================================================
          SELECTED CAMERA DETAILS
      ====================================================== */}

      {selectedCamera && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(4,minmax(0,1fr))",
            gap: "12px",
            marginBottom: "20px",
          }}
        >
          <DetailCard
            title="CAMERA ID"
            value={
              selectedCamera.camera_id
            }
          />

          <DetailCard
            title="SOURCE"
            value={
              selectedCamera.source_type ||
              "UNKNOWN"
            }
          />

          <DetailCard
            title="RESOLUTION"
            value={
              selectedCamera.width &&
              selectedCamera.height
                ? `${selectedCamera.width} × ${selectedCamera.height}`
                : "UNKNOWN"
            }
          />

          <DetailCard
            title="PROCESSING"
            value={
              selectedCamera.processed_frames ??
              0
            }
          />
        </div>
      )}

      {/* =====================================================
          FOOTER
      ====================================================== */}

      <div
        style={{
          borderTop:
            "1px solid #1e293b",
          paddingTop: "16px",
          color: "#64748b",
          fontSize: "12px",
          textAlign: "center",
        }}
      >
        IBVAP • INTELLIGENT BORDER
        VIDEO ANALYTICS PLATFORM
      </div>

      {/* =====================================================
          RESPONSIVE CSS
      ====================================================== */}

      <style>
        {`
          * {
            box-sizing: border-box;
          }

          button {
            transition:
              transform 0.15s ease,
              opacity 0.15s ease;
          }

          button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
          }

          @media (max-width: 1100px) {
            div[style*="repeat(4,minmax(0,1fr))"] {
              grid-template-columns:
                repeat(2,minmax(0,1fr)) !important;
            }
          }

          @media (max-width: 800px) {
            body {
              margin: 0;
            }

            div[style*="repeat(2,minmax(0,1fr))"] {
              grid-template-columns:
                1fr !important;
            }

            div[style*="height: 560px"] {
              height: 420px !important;
            }

            div[style*="height: 360px"] {
              height: 300px !important;
            }
          }

          @media (max-width: 600px) {
            div[style*="repeat(3,minmax(0,1fr))"] {
              grid-template-columns:
                1fr !important;
            }

            div[style*="repeat(4,minmax(0,1fr))"] {
              grid-template-columns:
                1fr !important;
            }
          }
        `}
      </style>
    </div>
  );
}

// =============================================================
// SMALL COMPONENTS
// =============================================================

function StatCard({
  title,
  value,
  accent,
}) {
  return (
    <div
      className="stat-card"
      style={{
        background:
          "#111827",
        border:
          "1px solid #25354d",
        borderRadius: "12px",
        padding: "18px",
      }}
    >
      <div
        style={{
          color: accent,
          fontWeight: 800,
          fontSize: "14px",
          marginBottom: "12px",
        }}
      >
        {title}
      </div>

      <div
        style={{
          fontSize: "28px",
          fontWeight: 900,
          color: "#f8fafc",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
}) {
  return (
    <div>
      <div
        style={{
          color: "#94a3b8",
          fontSize: "13px",
          marginBottom: "6px",
        }}
      >
        {label}
      </div>

      <div
        style={{
          color: "#f8fafc",
          fontSize: "21px",
          fontWeight: 900,
          wordBreak:
            "break-word",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function InfoBox({
  title,
  value,
}) {
  return (
    <div
      style={{
        background:
          "#111827",
        borderRadius: "8px",
        padding: "9px",
        minWidth: 0,
      }}
    >
      <div
        style={{
          color: "#64748b",
          fontSize: "10px",
          fontWeight: 800,
        }}
      >
        {title}
      </div>

      <div
        style={{
          marginTop: "4px",
          color: "#e2e8f0",
          fontSize: "12px",
          fontWeight: 800,
          overflow: "hidden",
          textOverflow:
            "ellipsis",
          whiteSpace:
            "nowrap",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function DetailCard({
  title,
  value,
}) {
  return (
    <div
      style={{
        background:
          "#111827",
        border:
          "1px solid #243244",
        borderRadius: "10px",
        padding: "14px",
      }}
    >
      <div
        style={{
          color: "#64748b",
          fontSize: "11px",
          fontWeight: 800,
        }}
      >
        {title}
      </div>

      <div
        style={{
          marginTop: "7px",
          color: "#e2e8f0",
          fontWeight: 900,
          fontSize: "14px",
          wordBreak:
            "break-word",
        }}
      >
        {value}
      </div>
    </div>
  );
}

export default App;