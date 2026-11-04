"""Integer/dictionary CPU fixtures only; no archived hardware observations."""

from analyze_native_probe import DOMAIN_NAMES
from typing import Any


def report():
    data: dict[str, Any] = dict(
        schema_version=1,
        kind="native_allocation_api_diagnostic",
        complete_workspace=False,
        hardware_validated=False,
        configured=True,
        init_called=True,
        ready=True,
        bounds_valid=True,
        write_complete=True,
        finalize_callback_called=True,
        create_status=0,
        valid_status=0,
        start_status=0,
        stop_status=0,
        callbacks_inflight=0,
        dropped_events=0,
        recursive_suppressed=0,
        unknown_events=0,
        missing_user_data=0,
        context_valid=1,
        metadata_truncated=False,
        buffered_services_used=False,
        buffer_flush_applicable=False,
        sdk_buffer_drop_count=None,
        filter_policy="all_mutators_global_and_all_APIs_in_or_entered_epoch",
        observed_callbacks=2,
        attempted_events=2,
        retained_events=2,
        max_events=100,
        filtered_nonmutator_events=0,
        init_ns=10,
        finalize_ns=500,
        stop_evidence=dict(
            invoked=True,
            completed=True,
            observation_window_closed=True,
            fini_skipped_second_stop=True,
            stop_result=0,
            drain_status=0,
            callbacks_after_stop=0,
            rejected_requests=0,
            origin=1,
            actual_sdk_stop_calls=1,
            begin_ns=450,
            end_ns=460,
            sdk_status_name="ROCPROFILER_STATUS_SUCCESS",
        ),
    )
    data["agent_inventory"] = dict(
        version=0,
        query_status=0,
        callback_count=1,
        reported_count=2,
        retained_count=2,
        max_agents=64,
        invalid=False,
        agents=[
            dict(
                handle=0,
                type=1,
                type_name="CPU",
                name="CPU fixture",
                product_name="CPU fixture",
                node_id=0,
                gpu_id=0,
                logical_node_id=0,
                logical_node_type_id=0,
            ),
            dict(
                handle=1,
                type=2,
                type_name="GPU",
                name="GPU fixture",
                product_name="GPU fixture",
                node_id=1,
                gpu_id=101,
                logical_node_id=1,
                logical_node_type_id=0,
            ),
        ],
    )
    data["domains"] = []
    for i, name in enumerate(DOMAIN_NAMES):
        op = dict(
            id=0,
            name="hipDeviceSynchronize" if i == 0 else "fixture_unused_api",
            flags=0,
            decoded_exit_status_type=1 if i == 0 else 0,
            enter_count=1 if i == 0 else 0,
            exit_count=1 if i == 0 else 0,
            other_phase_count=0,
            filtered_nonmutator_events=0,
        )
        data["domains"].append(
            dict(
                index=i, name=name, enumerate_status=0, configure_status=0, operation_count=1, operations=[op]
            )
        )
    data["epochs"] = [
        dict(
            id=1,
            label="cpu-fixture",
            begin_ns=100,
            end_ns=400,
            ended=True,
            successful_device_sync_exit_ns=300,
            api_events=2,
            mutator_events=0,
            unsupported_events=0,
            dropped_at_begin=0,
            dropped_at_end=0,
            recursive_at_begin=0,
            recursive_at_end=0,
        )
    ]
    data["events"] = [
        dict(
            seq=i,
            callback_seq=i,
            ns=100 + i * 100,
            tid=1,
            cid=1,
            epoch=1,
            domain=0,
            op=0,
            phase=i,
            flags=0,
            status_type=1 if i == 2 else 0,
            status=0 if i == 2 else None,
        )
        for i in (1, 2)
    ]
    return data


def report_v2(*, full=True, sentinel=True):
    data = report()
    data.update(
        probe_protocol_version=2,
        probe_protocol="native_allocation_api_probe_v2",
        retention_mode="all_callbacks_v1" if full else "mutators_and_epoch_v1",
        retain_all_option_valid=True,
        filter_policy="all_APIs_global" if full else "all_mutators_global_and_all_APIs_in_or_entered_epoch",
    )
    op = dict(
        id=0,
        name="ncclGetVersion",
        flags=512,
        decoded_exit_status_type=3,
        enter_count=int(sentinel),
        exit_count=int(sentinel),
        other_phase_count=0,
        filtered_nonmutator_events=0,
    )
    data["domains"].append(
        dict(
            index=8,
            name="RCCL_API",
            enumerate_status=0,
            configure_status=0,
            operation_count=1,
            operations=[op],
        )
    )
    if sentinel:
        for row in data["events"]:
            row["seq"] += 2
            row["callback_seq"] += 2
        calls = [
            dict(
                seq=i + 1,
                callback_seq=i + 1,
                ns=(50 if full else 150) + i,
                tid=1,
                cid=99,
                ancestor=0,
                epoch=0 if full else 1,
                domain=8,
                op=0,
                phase=i + 1,
                flags=512,
                status_type=3 if i else 0,
                status=0 if i else None,
            )
            for i in (0, 1)
        ]
        data["events"] = calls + data["events"]
        data.update(observed_callbacks=4, attempted_events=4, retained_events=4)
        if not full:
            data["epochs"][0]["api_events"] = 4
    return data
