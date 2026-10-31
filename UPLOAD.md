# GitHub upload and Pages setup

Confirmed repository: **googo2025/vtouch-simulation-benchmark**. This document records the publishing procedure. Keep it separate from the MagTouch reference project.

1. Extract the ZIP and review README.md, README.zh-CN.md and RELEASE_CHECKLIST.md.
2. Create the intended GitHub repository, or select the confirmed existing VTouch repository.
3. Upload/commit the contents of VTouch-Simulation-Benchmark/, not an enclosing duplicate folder.
4. In repository Settings → Pages, choose Deploy from a branch, branch main, folder /docs.
5. The project Pages URL will be https://googo2025.github.io/vtouch-simulation-benchmark/. The site is independent of the Tencent gateway.
6. Open English and Chinese pages; verify A1, A2, B1 and B2 and a CSV download. Record the resulting commit identity and actual Pages URL in the release record.

For a Git CLI upload after the destination is confirmed:

    git init -b main
    git add .
    git commit -m "Prepare VTouch A/B paper companion"
    git remote add origin https://github.com/googo2025/vtouch-simulation-benchmark.git
    git push -u origin main

Use normal credential-manager authentication. Never place a token in a URL, source file, command argument or README.

Official setup guide:
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

The README provides links and thumbnails; executable 3D interaction is served by Pages, not embedded as executable JavaScript in README.
