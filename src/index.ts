import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import axios from 'axios';
import { GoogleAuth } from 'google-auth-library';
import { VertexAI } from '@google-cloud/vertexai';

dotenv.config();

const app = express();
const port = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

const auth = new GoogleAuth();
const projectId = process.env.GCP_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT;
const region = process.env.REGION || 'us-central1';

// Initialize Vertex AI SDK client
const vertexAI = new VertexAI({ project: projectId, location: region });

/**
 * Helper to fetch a Google-signed OIDC ID Token from the metadata server.
 * Uses the Web Backend's URL as the target audience.
 * If running locally outside GCP, it falls back to a mock development token.
 */
async function getOidcToken(targetAudience: string): Promise<string> {
  try {
    const client = await auth.getIdTokenClient(targetAudience);
    const headers = await client.getRequestHeaders() as any;
    const authHeader = headers['Authorization'] || headers['authorization'];
    if (authHeader && authHeader.startsWith('Bearer ')) {
      return authHeader.substring(7);
    }
    throw new Error('Authorization header not returned by auth client');
  } catch (err) {
    console.log('[AI Worker] GCP Metadata Server OIDC token request failed (this is expected when running locally). Falling back to mock token.');
    return 'mock-local-development-system-token';
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

/**
 * POST /process
 * Triggered by Google Cloud Tasks.
 * Simulates heavy AI transcription and summarization of a file.
 * Returns results back to the Web Backend's callbackUrl.
 */
app.post('/process', async (req, res) => {
  const { fileId, userId, bucket, storagePath, callbackUrl } = req.body;

  if (!fileId || !userId || !bucket || !storagePath || !callbackUrl) {
    console.error('[AI Worker] Missing parameters in process request:', req.body);
    return res.status(400).json({ error: 'Bad Request: Missing required parameters' });
  }

  console.log(`[AI Worker] Starting AI processing for file: ${fileId} (User: ${userId})`);
  console.log(`[AI Worker] Target callback: ${callbackUrl}`);

  // Process synchronously to let Cloud Tasks track completion state
  try {
    // 1. Generate AI results using Vertex AI Gemini
    const isPdf = storagePath.toLowerCase().endsWith('.pdf');
    let summary = '';
    let transcription = '';

    // If running in local development mode without GCP credentials, fall back to mock AI data
    if (!projectId || process.env.NODE_ENV === 'development') {
      console.log('[AI Worker] Running in development mode or GCP_PROJECT_ID is missing. Simulating AI processing...');
      await new Promise((resolve) => setTimeout(resolve, 10000));
      
      if (isPdf) {
        summary = `### Executive Summary (Simulated)\n\nThis PDF document **"${fileId}"** was analyzed successfully using simulated Gemini intelligence.\n\n#### Key Findings:\n*   **Cloud Architecture**: Decoupling web servers from compute workers avoids connection terminations.\n*   **Server-Sent Events**: Delivers instant, multiplexed pushes directly to browsers under HTTP/2.\n*   **State Integrity**: Client states align automatically upon webhook database commits.`;
        transcription = 'N/A (PDF Document File)';
      } else {
        summary = `### Video Summary (Simulated)\n\nThis video was transcribed and analyzed successfully. Key discussion points:\n*   **Decoupled Workers**: Offloading Vertex AI requests ensures low-latency REST APIs for web users.\n*   **Cloud Tasks**: Provides rate-limiting (e.g. 5 concurrent dispatches) to protect backend resource constraints.\n*   **Heartbeat Pings**: Keeps connection sockets open across proxies under Google Cloud's GFE.`;
        transcription = `[00:01] Hello and welcome to OmniBrief AI.\n[00:05] Today we're configuring Server-Sent Events with Cloud Tasks.\n[00:10] The worker does the heavy processing and sends a callback to the backend when done.`;
      }
    } else {
      console.log(`[AI Worker] Invoking Vertex AI Gemini for gs://${bucket}/${storagePath}`);
      const mimeType = isPdf ? 'application/pdf' : 'video/mp4';
      const fileUri = `gs://${bucket}/${storagePath}`;

      const generativeModel = vertexAI.getGenerativeModel({
        model: 'gemini-1.5-flash',
      });

      const filePart = {
        fileData: {
          fileUri: fileUri,
          mimeType: mimeType,
        },
      };

      const promptText = isPdf
        ? "Analyze this PDF document. Provide a comprehensive summary in clear Markdown formatting, explaining the key points, main conclusions, and highlights."
        : "Analyze this video file. Provide two sections in your response:\n1. Summary: A detailed summary of the video content.\n2. Transcript: A chronological transcription of the speech in the video with timestamps.\nFormat your response in Markdown.";

      const promptPart = {
        text: promptText,
      };

      const request = {
        contents: [{ role: 'user', parts: [filePart, promptPart] }],
      };

      const responseResult = await generativeModel.generateContent(request);
      const response = await responseResult.response;
      const responseText = response.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!responseText) {
        throw new Error('Empty response received from Gemini');
      }

      summary = responseText;
      transcription = 'N/A';

      if (!isPdf) {
        // Attempt to extract transcription block from Gemini markdown response
        const splitIndex = responseText.toLowerCase().indexOf('transcript:');
        const alternateSplitIndex = responseText.toLowerCase().indexOf('## transcript');
        const targetIndex = splitIndex !== -1 ? splitIndex : (alternateSplitIndex !== -1 ? alternateSplitIndex : -1);
        
        if (targetIndex !== -1) {
          summary = responseText.substring(0, targetIndex).trim();
          transcription = responseText.substring(targetIndex).trim();
        }
      }
    }

    // 3. Obtain Google OIDC token for callback authorization
    // The target audience is the base URL of the callback (i.e. our backend service URL)
    const audience = new URL(callbackUrl).origin;
    console.log(`[AI Worker] Fetching OIDC token for audience: ${audience}`);
    const token = await getOidcToken(audience);

    // 4. Secure callback to Web Backend
    console.log(`[AI Worker] Sending callback results to Web Backend...`);
    const response = await axios.post(
      callbackUrl,
      {
        userId,
        summary,
        transcription,
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      }
    );

    console.log(`[AI Worker] Callback completed successfully. Status: ${response.status}`);
    return res.status(200).json({
      status: 'success',
      message: 'AI Processing and Callback completed successfully',
      fileId,
    });

  } catch (error: any) {
    console.error(`[AI Worker] Error during processing or callback:`, error.message || error);
    // Return a 500 error so that Cloud Tasks automatically retries the task later
    return res.status(500).json({
      status: 'error',
      message: error.message || 'Internal processing error',
    });
  }
});

app.listen(port, () => {
  console.log(`[AI Worker] Server is running on port: ${port}`);
});
