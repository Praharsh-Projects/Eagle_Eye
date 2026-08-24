import type {
  AnswerEnvelope,
  CapabilityResponse,
  ExportRequest,
  ExportResponse,
  FeedbackResponse,
  QueryMode,
  QueryRequestPayload,
} from "./types";

const API_ROOT = (import.meta.env.VITE_API_BASE_URL || "/api/v2").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = "The intelligence service could not complete this request.";
    try {
      const payload = (await response.json()) as { detail?: string; message?: string };
      message = payload.detail || payload.message || message;
    } catch {
      // Preserve the safe fallback when an intermediary returns non-JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function submitQuery(query: QueryRequestPayload): Promise<AnswerEnvelope>;
/** @deprecated Use the QueryRequest overload for reproducible analyses. */
export function submitQuery(question: string, conversationId?: string): Promise<AnswerEnvelope>;
export async function submitQuery(
  queryOrQuestion: QueryRequestPayload | string,
  conversationId?: string,
): Promise<AnswerEnvelope> {
  const payload: QueryRequestPayload =
    typeof queryOrQuestion === "string"
      ? {
          question: queryOrQuestion,
          conversation_id: conversationId,
        }
      : queryOrQuestion;
  const envelope = await request<
    Omit<AnswerEnvelope, "mode"> & { mode?: QueryMode | "general" }
  >("/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  // Transitional API deployments returned `general`; normalize that legacy
  // value at the boundary while the UI uses the canonical `general_chat` mode.
  return {
    ...envelope,
    mode: envelope.mode === "general" ? "general_chat" : envelope.mode || "app_help",
  };
}

export async function loadCapabilities(): Promise<CapabilityResponse | null> {
  try {
    return await request<CapabilityResponse>("/capabilities");
  } catch {
    return null;
  }
}

export async function exportDataset(payload: ExportRequest): Promise<ExportResponse> {
  return request<ExportResponse>("/exports", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function reportIssue(
  traceId: string,
  question: string,
  detail: string,
): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/feedback", {
    method: "POST",
    body: JSON.stringify({ trace_id: traceId, prompt: question, note: detail }),
  });
}
